#!/usr/bin/env python3
"""
Moonshine MLX Benchmark Suite

Measures:
  1. Cold start (model load + first transcription)
  2. Steady-state streaming latency (per-chunk after warmup)
  3. Peak memory usage
  4. Tokens per second
  5. Total processing time at various audio lengths
  6. Model parameter count and size on disk

Usage:
  python benchmarks/bench_suite.py [--models tiny-fp16,small-8bit,...] [--swift-only] [--python-only]

Output:
  benchmarks/results/suite_<timestamp>.json
  Summary table to stdout
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

# ── Paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SWIFT_CLI = os.path.join(PROJECT_DIR, ".build", "arm64-apple-macosx", "release", "moonshine-mlx")
MODELS_DIR = "/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex/models"
MLX_AUDIO_DIR = "/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

AUDIO_DIR = os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart")
SHORT_AUDIO = os.path.join(AUDIO_DIR, "00000001-Genesis-1:1.wav")  # ~6.6s

# Audio files at various lengths (created by the script if missing)
LONG_AUDIO_SOURCE = "/Users/kylehowells/Movies/YouTube/How AI Datacenters Eat the World [dhqoTku-HAA].mp3"
AUDIO_LENGTHS = [6, 30, 60, 600]  # seconds

# ── Model Registry ───────────────────────────────────────────────────

MODELS = {
    # name: (path_or_repo, params_millions, description)
    "tiny-fp32":   ("UsefulSensors/moonshine-streaming-tiny",  34,  "Tiny fp32 (HuggingFace)"),
    "tiny-fp16":   (f"{MODELS_DIR}/moonshine-streaming-tiny-fp16",   34,  "Tiny fp16"),
    "tiny-8bit":   (f"{MODELS_DIR}/moonshine-streaming-tiny-8bit",   34,  "Tiny 8-bit quantized"),
    "tiny-4bit":   (f"{MODELS_DIR}/moonshine-streaming-tiny-4bit",   34,  "Tiny 4-bit quantized"),
    "small-fp16":  (f"{MODELS_DIR}/moonshine-streaming-small-fp16",  123, "Small fp16"),
    "small-8bit":  (f"{MODELS_DIR}/moonshine-streaming-small-8bit",  123, "Small 8-bit quantized"),
    "small-4bit":  (f"{MODELS_DIR}/moonshine-streaming-small-4bit",  123, "Small 4-bit quantized"),
    "medium-fp16": (f"{MODELS_DIR}/moonshine-streaming-medium-fp16", 245, "Medium fp16"),
    "medium-8bit": (f"{MODELS_DIR}/moonshine-streaming-medium-8bit", 245, "Medium 8-bit quantized"),
    "medium-4bit": (f"{MODELS_DIR}/moonshine-streaming-medium-4bit", 245, "Medium 4-bit quantized"),
}

# ── Helpers ──────────────────────────────────────────────────────────

def ensure_audio_files():
    """Create test audio files at various lengths if they don't exist."""
    os.makedirs("/tmp", exist_ok=True)
    files = {}
    for dur in AUDIO_LENGTHS:
        if dur == 6:
            files[dur] = SHORT_AUDIO
            continue
        path = f"/tmp/moonshine_bench_{dur}s.wav"
        if not os.path.exists(path):
            if not os.path.exists(LONG_AUDIO_SOURCE):
                print(f"  Warning: {LONG_AUDIO_SOURCE} not found, skipping {dur}s audio")
                continue
            subprocess.run([
                "ffmpeg", "-y", "-i", LONG_AUDIO_SOURCE,
                "-t", str(dur), "-ac", "1", "-ar", "16000", "-f", "wav", path
            ], capture_output=True)
        files[dur] = path
    return files


def run_swift(model_path, audio_path, extra_args=None):
    """Run the Swift CLI and parse JSON output. Returns dict or None on failure."""
    cmd = [SWIFT_CLI, audio_path, "--model", model_path, "--json"]
    if extra_args:
        cmd.extend(extra_args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except Exception:
        return None


def model_size_on_disk(model_path):
    """Get total size of safetensors files in MB."""
    total = 0
    if os.path.isdir(model_path):
        for f in os.listdir(model_path):
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(model_path, f))
    return round(total / 1e6, 1)


# ── Benchmark Functions ──────────────────────────────────────────────

def bench_cold_start(model_name, model_path, audio_path):
    """Measure cold start: wall time for a fresh CLI invocation (includes model load)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = subprocess.run(
            [SWIFT_CLI, audio_path, "--model", model_path, "--json"],
            capture_output=True, text=True, timeout=300
        )
        wall = time.perf_counter() - t0
        if r.returncode == 0:
            d = json.loads(r.stdout)
            times.append({
                "wall_time_s": round(wall, 3),
                "inference_time_s": d.get("transcription_time_s", 0),
                "model_load_time_s": d.get("model_load_time_s", 0),
                "tokens": d.get("generation_tokens", 0),
                "peak_memory_mb": d.get("peak_memory_mb", 0),
            })
    if not times:
        return None
    # Report median wall time
    times.sort(key=lambda x: x["wall_time_s"])
    return times[len(times) // 2]


def bench_steady_state(model_name, model_path, audio_path):
    """Measure steady-state: run 5 times, report last 3 (warmed)."""
    results = []
    for i in range(5):
        d = run_swift(model_path, audio_path)
        if d and i >= 2:  # skip first 2 as warmup
            results.append({
                "transcription_time_s": d.get("transcription_time_s", 0),
                "tokens": d.get("generation_tokens", 0),
                "tokens_per_second": d.get("tokens_per_second", 0),
                "peak_memory_mb": d.get("peak_memory_mb", 0),
            })
    if not results:
        return None
    # Average the warmed runs
    avg = {}
    for key in results[0]:
        avg[key] = round(sum(r[key] for r in results) / len(results), 2)
    return avg


def bench_audio_lengths(model_name, model_path, audio_files):
    """Measure total processing time at various audio lengths (warmed)."""
    results = {}
    for dur, path in sorted(audio_files.items()):
        if not os.path.exists(path):
            continue
        # Warmup
        run_swift(model_path, path)
        # Timed run (best of 3)
        times = []
        for _ in range(3):
            d = run_swift(model_path, path)
            if d:
                times.append({
                    "audio_duration_s": dur,
                    "transcription_time_s": d.get("transcription_time_s", 0),
                    "tokens": d.get("generation_tokens", 0),
                    "tokens_per_second": d.get("tokens_per_second", 0),
                    "peak_memory_mb": d.get("peak_memory_mb", 0),
                    "real_time_factor": round(d.get("transcription_time_s", 0) / dur, 4) if dur > 0 else 0,
                })
        if times:
            # Report best (lowest transcription time)
            times.sort(key=lambda x: x["transcription_time_s"])
            results[dur] = times[0]
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Moonshine MLX Benchmark Suite")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip long audio tests (only 6s and 30s)")
    args = parser.parse_args()

    if not os.path.exists(SWIFT_CLI):
        print(f"Error: Swift CLI not found at {SWIFT_CLI}")
        print("Run: cd moonshine-mlx && swift build -c release")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODELS.keys())

    # Prepare audio files
    audio_files = ensure_audio_files()
    if args.quick:
        audio_files = {k: v for k, v in audio_files.items() if k <= 30}

    print(f"Benchmarking {len(model_names)} models on {len(audio_files)} audio lengths\n")

    all_results = []

    for model_name in model_names:
        if model_name not in MODELS:
            print(f"  Unknown model: {model_name}, skipping")
            continue

        model_path, params_m, description = MODELS[model_name]

        # Check model exists
        if not os.path.exists(model_path) and "/" in model_path and not model_path.startswith("UsefulSensors"):
            print(f"  {model_name}: model not found at {model_path}, skipping")
            continue

        print(f"{'='*60}")
        print(f"  {model_name} ({description})")
        print(f"{'='*60}")

        disk_mb = model_size_on_disk(model_path) if os.path.isdir(model_path) else 0

        # 1. Cold start
        print(f"  Cold start...", end="", flush=True)
        cold = bench_cold_start(model_name, model_path, SHORT_AUDIO)
        if cold:
            print(f" {cold['wall_time_s']:.2f}s wall, {cold['inference_time_s']*1000:.0f}ms inference")
        else:
            print(" FAILED")

        # 2. Steady state (6.6s audio)
        print(f"  Steady state (6.6s)...", end="", flush=True)
        steady = bench_steady_state(model_name, model_path, SHORT_AUDIO)
        if steady:
            print(f" {steady['transcription_time_s']*1000:.0f}ms, {steady['tokens_per_second']:.0f} tok/s, {steady['peak_memory_mb']:.0f}MB")
        else:
            print(" FAILED")

        # 3. Audio lengths
        print(f"  Audio lengths...", end="", flush=True)
        lengths = bench_audio_lengths(model_name, model_path, audio_files)
        for dur, r in sorted(lengths.items()):
            print(f"\n    {dur:>4d}s: {r['transcription_time_s']:.3f}s, {r['tokens']:3d} tok, {r['tokens_per_second']:.0f} tok/s, RTF {r['real_time_factor']:.4f}x", end="")
        print()

        result = {
            "model": model_name,
            "description": description,
            "parameters_millions": params_m,
            "model_size_mb": disk_mb,
            "cold_start": cold,
            "steady_state_6s": steady,
            "audio_lengths": lengths,
        }
        all_results.append(result)
        print()

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "benchmark": "moonshine-mlx-suite",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "swift_cli": SWIFT_CLI,
        "results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, f"suite_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Also save as latest
    latest_path = os.path.join(RESULTS_DIR, "suite_latest.json")
    with open(latest_path, "w") as f:
        json.dump(out, f, indent=2)

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Model':<14} {'Params':>6} {'Disk':>6} {'Cold':>7} {'Warm 6s':>8} {'Tok/s':>6} {'Mem':>7}"
          f" {'30s':>7} {'60s':>7} {'10m':>7}")
    print("-" * 100)

    for r in all_results:
        cold_s = f"{r['cold_start']['wall_time_s']:.2f}s" if r.get("cold_start") else "—"
        warm_ms = f"{r['steady_state_6s']['transcription_time_s']*1000:.0f}ms" if r.get("steady_state_6s") else "—"
        tps = f"{r['steady_state_6s']['tokens_per_second']:.0f}" if r.get("steady_state_6s") else "—"
        mem = f"{r['steady_state_6s']['peak_memory_mb']:.0f}MB" if r.get("steady_state_6s") else "—"

        t30 = t60 = t600 = "—"
        al = r.get("audio_lengths", {})
        if 30 in al: t30 = f"{al[30]['transcription_time_s']:.2f}s"
        if 60 in al: t60 = f"{al[60]['transcription_time_s']:.2f}s"
        if 600 in al: t600 = f"{al[600]['transcription_time_s']:.2f}s"

        print(f"{r['model']:<14} {r['parameters_millions']:>4}M {r.get('model_size_mb',0):>5.0f}M"
              f" {cold_s:>7} {warm_ms:>8} {tps:>6} {mem:>7}"
              f" {t30:>7} {t60:>7} {t600:>7}")

    print(f"\nResults saved to: {out_path}")
    print(f"Latest link: {latest_path}")


if __name__ == "__main__":
    main()
