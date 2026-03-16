#!/usr/bin/env python3
"""
Combined benchmark: Python MLX vs Swift MLX for Moonshine V2 Streaming.

Measures: model load time, transcription latency, tokens/sec, peak memory, RTF, WER.
Runs both implementations on the same audio files and saves results to JSON.
"""

import gc
import json
import os
import subprocess
import sys
import time

MLX_AUDIO_DIR = "/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex"
SWIFT_CLI = "/Users/kylehowells/Developer/Github/moonshine-mlx/.build/release/moonshine-mlx"
sys.path.insert(0, MLX_AUDIO_DIR)

# ── Configuration ────────────────────────────────────────────────────

AUDIO_FILES = [
    ("genesis_1_1",
     os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav"),
     "In the beginning, God created the heaven and the earth."),
    ("genesis_1_2",
     os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000002-Genesis-1:2.wav"),
     "And the earth was without form and void, and darkness was on the face of the deep, and the spirit of God moved on the face of the waters."),
    ("genesis_1_3",
     os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000003-Genesis-1:3.wav"),
     "And God said, Let there be light, and there was light."),
]

MODELS = [
    ("tiny", "UsefulSensors/moonshine-streaming-tiny"),
    ("small", "UsefulSensors/moonshine-streaming-small"),
]

NUM_WARMUP = 1
NUM_RUNS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ── WER ──────────────────────────────────────────────────────────────

def word_error_rate(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    r, h = len(ref), len(hyp)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[r][h] / max(r, 1)


def strip_verse_prefix(text):
    import re
    return re.sub(r'^Genesis \d+:\d+\s*', '', text).strip()


# ── Python Benchmark ─────────────────────────────────────────────────

def bench_python(model_label, model_path, audio_label, audio_path, reference):
    import mlx.core as mx
    from mlx_audio.utils import get_model_path, load_config, load_weights
    from mlx_audio.stt.models.moonshine_streaming import Model, ModelConfig
    from mlx_audio.stt.utils import load_audio
    from mlx.utils import tree_reduce

    print(f"  [python] Loading {model_label}...")
    gc.collect(); mx.reset_peak_memory(); mx.synchronize()

    t0 = time.perf_counter()
    path = get_model_path(model_path)
    config = load_config(path)
    mc = ModelConfig.from_dict(config)
    model = Model(mc)
    weights = load_weights(path)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters()); model.eval()
    Model.post_load_hook(model, path)
    mx.synchronize()
    load_time = time.perf_counter() - t0
    model_bytes = tree_reduce(lambda a, x: a + x.nbytes if isinstance(x, mx.array) else a, model, 0)

    audio = load_audio(audio_path, sr=model.sample_rate)
    dur = audio.shape[-1] / model.sample_rate

    # Warmup
    for _ in range(NUM_WARMUP):
        mx.reset_peak_memory()
        model.generate(audio, verbose=False)
        mx.synchronize()

    # Timed runs
    runs = []
    for i in range(NUM_RUNS):
        gc.collect(); mx.synchronize(); mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = model.generate(audio, verbose=False)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        peak = mx.get_peak_memory()
        text = out.text.strip()
        toks = out.generation_tokens
        tps = toks / elapsed if elapsed > 0 else 0
        rtf = elapsed / dur
        hyp = strip_verse_prefix(text)
        wer = word_error_rate(reference, hyp)
        runs.append({"run": i+1, "time_s": round(elapsed, 4), "tokens": toks,
                      "tps": round(tps, 1), "rtf": round(rtf, 4),
                      "peak_mb": round(peak/1e6, 1), "wer": round(wer, 4), "text": text})
        print(f"    run {i+1}: {elapsed:.3f}s | {toks} tok | {tps:.0f} tok/s | WER {wer:.0%}")

    del model; gc.collect(); mx.synchronize()
    return {
        "impl": "python-mlx", "model": model_label, "audio": audio_label,
        "audio_duration_s": round(dur, 3), "model_mb": round(model_bytes/1e6, 1),
        "load_time_s": round(load_time, 4), "runs": runs,
        "avg_time_s": round(sum(r["time_s"] for r in runs)/len(runs), 4),
        "avg_tps": round(sum(r["tps"] for r in runs)/len(runs), 1),
        "avg_rtf": round(sum(r["rtf"] for r in runs)/len(runs), 4),
        "avg_peak_mb": round(sum(r["peak_mb"] for r in runs)/len(runs), 1),
        "avg_wer": round(sum(r["wer"] for r in runs)/len(runs), 4),
    }


# ── Swift Benchmark ──────────────────────────────────────────────────

def bench_swift(model_label, model_path, audio_label, audio_path, reference):
    print(f"  [swift]  Loading {model_label}...")
    if not os.path.exists(SWIFT_CLI):
        print(f"    ERROR: Swift CLI not found at {SWIFT_CLI}")
        return {"impl": "swift-mlx", "model": model_label, "audio": audio_label, "error": "CLI not found"}

    # Warmup run
    subprocess.run([SWIFT_CLI, audio_path, "--model", model_path, "--json"],
                   capture_output=True, timeout=120)

    runs = []
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        result = subprocess.run(
            [SWIFT_CLI, audio_path, "--model", model_path, "--json"],
            capture_output=True, text=True, timeout=120
        )
        wall_time = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"    run {i+1}: ERROR - {result.stderr[:200]}")
            runs.append({"run": i+1, "error": result.stderr[:200]})
            continue

        try:
            metrics = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"    run {i+1}: JSON parse error - stdout: {result.stdout[:200]}")
            runs.append({"run": i+1, "error": "JSON parse error"})
            continue

        text = metrics.get("text", "")
        toks = metrics.get("generation_tokens", 0)
        t = metrics.get("transcription_time_s", wall_time)
        tps = metrics.get("tokens_per_second", 0)
        rtf = metrics.get("real_time_factor", 0)
        peak = metrics.get("peak_memory_mb", 0)
        hyp = strip_verse_prefix(text)
        wer = word_error_rate(reference, hyp)

        runs.append({"run": i+1, "time_s": round(t, 4), "tokens": toks,
                      "tps": round(tps, 1), "rtf": round(rtf, 4),
                      "peak_mb": round(peak, 1), "wer": round(wer, 4), "text": text,
                      "wall_time_s": round(wall_time, 4),
                      "load_time_s": round(metrics.get("model_load_time_s", 0), 4)})
        print(f"    run {i+1}: {t:.3f}s | {toks} tok | {tps:.0f} tok/s | WER {wer:.0%} (wall {wall_time:.2f}s)")

    valid = [r for r in runs if "error" not in r]
    if not valid:
        return {"impl": "swift-mlx", "model": model_label, "audio": audio_label, "error": "all runs failed", "runs": runs}

    return {
        "impl": "swift-mlx", "model": model_label, "audio": audio_label,
        "audio_duration_s": valid[0].get("time_s", 0) / max(valid[0].get("rtf", 1), 0.001),
        "runs": runs,
        "avg_time_s": round(sum(r["time_s"] for r in valid)/len(valid), 4),
        "avg_tps": round(sum(r["tps"] for r in valid)/len(valid), 1),
        "avg_rtf": round(sum(r["rtf"] for r in valid)/len(valid), 4),
        "avg_peak_mb": round(sum(r["peak_mb"] for r in valid)/len(valid), 1),
        "avg_wer": round(sum(r["wer"] for r in valid)/len(valid), 4),
        "avg_wall_time_s": round(sum(r.get("wall_time_s", r["time_s"]) for r in valid)/len(valid), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for model_label, model_path in MODELS:
        for audio_label, audio_path, ref in AUDIO_FILES:
            print(f"\n{'='*60}")
            print(f"  {model_label} | {audio_label}")
            print(f"{'='*60}")

            try:
                py = bench_python(model_label, model_path, audio_label, audio_path, ref)
                all_results.append(py)
            except Exception as e:
                print(f"  [python] ERROR: {e}")
                all_results.append({"impl": "python-mlx", "model": model_label, "audio": audio_label, "error": str(e)})

            try:
                sw = bench_swift(model_label, model_path, audio_label, audio_path, ref)
                all_results.append(sw)
            except Exception as e:
                print(f"  [swift]  ERROR: {e}")
                all_results.append({"impl": "swift-mlx", "model": model_label, "audio": audio_label, "error": str(e)})

    out_path = os.path.join(OUTPUT_DIR, "benchmark_comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "moonshine-mlx-python-vs-swift",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": "Apple Silicon (MLX)",
            "num_warmup": NUM_WARMUP, "num_runs": NUM_RUNS,
            "results": all_results,
        }, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Impl':<12} {'Model':<8} {'Audio':<14} {'Avg Time':>10} {'Tok/s':>8} {'RTF':>8} {'Mem MB':>8} {'WER':>6}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r.get('impl','?'):<12} {r.get('model','?'):<8} {r.get('audio','?'):<14} {'ERROR':>10}")
            continue
        print(f"{r['impl']:<12} {r['model']:<8} {r['audio']:<14} "
              f"{r['avg_time_s']:>9.3f}s {r['avg_tps']:>7.0f} {r['avg_rtf']:>7.4f}x "
              f"{r['avg_peak_mb']:>7.0f} {r['avg_wer']:>5.0%}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
