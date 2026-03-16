#!/usr/bin/env python3
"""
Benchmark Python MLX Moonshine V2 Streaming models.

Measures: model load time, transcription latency, tokens/sec, peak memory, RTF, WER.
Outputs JSON results for comparison with the Swift implementation.

Note: Uses direct V2 model loading to work around a model-type resolution bug
in mlx-audio's base_load_model where "moonshine" in repo name matches V1 module.
"""

import gc
import json
import os
import sys
import time

MLX_AUDIO_DIR = "/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex"
sys.path.insert(0, MLX_AUDIO_DIR)

import mlx.core as mx
from mlx_audio.utils import get_model_path, load_config, load_weights
from mlx_audio.stt.models.moonshine_streaming import Model, ModelConfig
from mlx_audio.stt.utils import load_audio

# ── Configuration ────────────────────────────────────────────────────

AUDIO_FILES = [
    (
        "genesis_1_1",
        os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav"),
        "In the beginning, God created the heaven and the earth.",
    ),
    (
        "genesis_1_2",
        os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000002-Genesis-1:2.wav"),
        "And the earth was without form and void, and darkness was on the face of the deep, and the spirit of God moved on the face of the waters.",
    ),
    (
        "genesis_1_3",
        os.path.join(MLX_AUDIO_DIR, "examples/bible-audiobook/audios/bible-akjv/af_heart/00000003-Genesis-1:3.wav"),
        "And God said, Let there be light, and there was light.",
    ),
]

MODELS = [
    ("tiny", "UsefulSensors/moonshine-streaming-tiny"),
    ("small", "UsefulSensors/moonshine-streaming-small"),
]

NUM_WARMUP = 1
NUM_RUNS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── WER ──────────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    r, h = len(ref), len(hyp)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[r][h] / max(r, 1)


# ── Helpers ──────────────────────────────────────────────────────────

def get_model_size_bytes(model):
    from mlx.utils import tree_reduce
    return tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )


def load_v2_model(model_path_str):
    """Load Moonshine V2 model directly (bypasses buggy model-type resolution)."""
    path = get_model_path(model_path_str)
    config = load_config(path)
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)
    weights = load_weights(path)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    Model.post_load_hook(model, path)
    return model


def benchmark_model(model_label, model_path, audio_label, audio_path, reference_text):
    """Run a full benchmark for one model + one audio file."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_label} ({model_path})")
    print(f"  Audio: {audio_label}")
    print(f"{'='*60}")

    # ── Load model ──
    gc.collect()
    mx.reset_peak_memory()
    mx.synchronize()

    load_start = time.perf_counter()
    model = load_v2_model(model_path)
    mx.synchronize()
    load_time = time.perf_counter() - load_start
    model_bytes = get_model_size_bytes(model)

    print(f"  Load time:     {load_time:.3f}s")
    print(f"  Model size:    {model_bytes / 1e6:.1f} MB")

    # ── Load audio ──
    audio = load_audio(audio_path, sr=model.sample_rate)
    audio_duration = audio.shape[-1] / model.sample_rate
    print(f"  Audio duration: {audio_duration:.2f}s")

    # ── Warmup ──
    print(f"  Warming up ({NUM_WARMUP} run)...")
    for _ in range(NUM_WARMUP):
        mx.reset_peak_memory()
        _ = model.generate(audio, verbose=False)
        mx.synchronize()

    # ── Timed runs ──
    results = []
    for i in range(NUM_RUNS):
        gc.collect()
        mx.synchronize()
        mx.reset_peak_memory()

        run_start = time.perf_counter()
        output = model.generate(audio, verbose=False)
        mx.synchronize()
        run_time = time.perf_counter() - run_start
        peak_mem = mx.get_peak_memory()

        gen_tokens = output.generation_tokens if hasattr(output, "generation_tokens") else 0
        text = output.text.strip() if hasattr(output, "text") else ""
        tps = gen_tokens / run_time if run_time > 0 else 0
        rtf = run_time / audio_duration if audio_duration > 0 else 0

        # Strip prefix like "Genesis 1:1 " before WER
        hyp = text
        for prefix in ["Genesis 1:1 ", "Genesis 1:2 ", "Genesis 1:3 "]:
            if hyp.startswith(prefix):
                hyp = hyp[len(prefix):]
                break
        wer = word_error_rate(reference_text, hyp)

        results.append({
            "run": i + 1,
            "transcription_time_s": round(run_time, 4),
            "generation_tokens": gen_tokens,
            "tokens_per_second": round(tps, 2),
            "real_time_factor": round(rtf, 4),
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": round(peak_mem / 1e6, 1),
            "text": text,
            "wer": round(wer, 4),
        })
        print(f"  Run {i+1}: {run_time:.3f}s | {gen_tokens} tok | {tps:.1f} tok/s | RTF {rtf:.3f} | WER {wer:.2%} | mem {peak_mem/1e6:.1f}MB")
        if i == 0:
            print(f"         Text: \"{text}\"")

    # ── Aggregate ──
    avg_time = sum(r["transcription_time_s"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    avg_rtf = sum(r["real_time_factor"] for r in results) / len(results)
    avg_mem = sum(r["peak_memory_mb"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    best_time = min(r["transcription_time_s"] for r in results)

    del model
    gc.collect()
    mx.synchronize()

    return {
        "implementation": "python-mlx",
        "model_label": model_label,
        "model_path": model_path,
        "audio_label": audio_label,
        "audio_file": os.path.basename(audio_path),
        "audio_duration_s": round(audio_duration, 3),
        "reference_text": reference_text,
        "model_size_mb": round(model_bytes / 1e6, 1),
        "model_load_time_s": round(load_time, 4),
        "num_warmup": NUM_WARMUP,
        "num_runs": NUM_RUNS,
        "runs": results,
        "avg_transcription_time_s": round(avg_time, 4),
        "best_transcription_time_s": round(best_time, 4),
        "avg_tokens_per_second": round(avg_tps, 2),
        "avg_real_time_factor": round(avg_rtf, 4),
        "avg_peak_memory_mb": round(avg_mem, 1),
        "avg_wer": round(avg_wer, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for model_label, model_path in MODELS:
        for audio_label, audio_path, reference_text in AUDIO_FILES:
            try:
                result = benchmark_model(model_label, model_path, audio_label, audio_path, reference_text)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "implementation": "python-mlx",
                    "model_label": model_label,
                    "audio_label": audio_label,
                    "error": str(e),
                })

    out_path = os.path.join(OUTPUT_DIR, "benchmark_python_mlx.json")
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "moonshine-mlx-comparison",
            "implementation": "python-mlx",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": "Apple Silicon (MLX)",
            "results": all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
