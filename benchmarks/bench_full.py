#!/usr/bin/env python3
"""
Full benchmark: Python MLX vs Swift MLX on short + long audio.
"""

import gc, json, os, re, subprocess, sys, time

MLX_AUDIO_DIR = "/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex"
SWIFT_CLI = "/Users/kylehowells/Developer/Github/moonshine-mlx/.build/release/moonshine-mlx"
sys.path.insert(0, MLX_AUDIO_DIR)

AUDIO_FILES = [
    ("genesis_1_1 (6.6s)",
     f"{MLX_AUDIO_DIR}/examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav",
     "In the beginning, God created the heaven and the earth."),
    ("genesis_1_2 (13.2s)",
     f"{MLX_AUDIO_DIR}/examples/bible-audiobook/audios/bible-akjv/af_heart/00000002-Genesis-1:2.wav",
     "And the earth was without form and void, and darkness was on the face of the deep, and the spirit of God moved on the face of the waters."),
    ("datacenter_30s",
     "/tmp/bench_30s.wav", None),
    ("datacenter_60s",
     "/tmp/bench_60s.wav", None),
    ("datacenter_120s",
     "/tmp/bench_120s.wav", None),
]

MODELS = [
    ("tiny", "UsefulSensors/moonshine-streaming-tiny"),
    ("small", "UsefulSensors/moonshine-streaming-small"),
]

NUM_WARMUP = 1
NUM_RUNS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def wer(ref, hyp):
    if ref is None: return None
    r, h = ref.lower().split(), hyp.lower().split()
    rn, hn = len(r), len(h)
    d = [[0]*(hn+1) for _ in range(rn+1)]
    for i in range(rn+1): d[i][0] = i
    for j in range(hn+1): d[0][j] = j
    for i in range(1,rn+1):
        for j in range(1,hn+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[rn][hn] / max(rn,1)

def strip_prefix(t):
    return re.sub(r'^Genesis \d+:\d+\s*', '', t).strip()


def bench_python(model_label, model_path, audio_label, audio_path, ref):
    import mlx.core as mx
    from mlx_audio.utils import get_model_path, load_config, load_weights
    from mlx_audio.stt.models.moonshine_streaming import Model, ModelConfig
    from mlx_audio.stt.utils import load_audio
    from mlx.utils import tree_reduce

    gc.collect(); mx.reset_peak_memory(); mx.synchronize()
    t0 = time.perf_counter()
    path = get_model_path(model_path)
    config = load_config(path)
    model = Model(ModelConfig.from_dict(config))
    weights = load_weights(path)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters()); model.eval()
    Model.post_load_hook(model, path)
    mx.synchronize()
    load_time = time.perf_counter() - t0
    model_mb = tree_reduce(lambda a,x: a+x.nbytes if isinstance(x,mx.array) else a, model, 0) / 1e6

    audio = load_audio(audio_path, sr=model.sample_rate)
    dur = audio.shape[-1] / model.sample_rate

    for _ in range(NUM_WARMUP):
        mx.reset_peak_memory()
        model.generate(audio, verbose=False); mx.synchronize()

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
        w = wer(ref, strip_prefix(text)) if ref else None
        runs.append({"time_s": round(elapsed,4), "tokens": toks,
                      "tps": round(toks/elapsed,1) if elapsed>0 else 0,
                      "peak_mb": round(peak/1e6,1), "wer": w,
                      "text": text[:200]})

    del model; gc.collect(); mx.synchronize()
    avg_t = sum(r["time_s"] for r in runs)/len(runs)
    avg_tps = sum(r["tps"] for r in runs)/len(runs)
    return {"impl":"python", "model":model_label, "audio":audio_label,
            "dur_s":round(dur,1), "model_mb":round(model_mb,1), "load_s":round(load_time,2),
            "avg_time_s":round(avg_t,4), "avg_tps":round(avg_tps,1),
            "avg_peak_mb":round(sum(r["peak_mb"] for r in runs)/len(runs),1),
            "runs":runs}


def bench_swift(model_label, model_path, audio_label, audio_path, ref):
    if not os.path.exists(SWIFT_CLI):
        return {"impl":"swift", "model":model_label, "audio":audio_label, "error":"CLI not found"}

    # Warmup
    subprocess.run([SWIFT_CLI, audio_path, "--model", model_path, "--json"],
                   capture_output=True, timeout=300)

    runs = []
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        r = subprocess.run([SWIFT_CLI, audio_path, "--model", model_path, "--json"],
                           capture_output=True, text=True, timeout=300)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            runs.append({"error": r.stderr[:200]}); continue
        try:
            m = json.loads(r.stdout)
        except: runs.append({"error":"json parse"}); continue
        text = m.get("text","")
        toks = m.get("generation_tokens",0)
        t = m.get("transcription_time_s", wall)
        w = wer(ref, strip_prefix(text)) if ref else None
        runs.append({"time_s":round(t,4), "tokens":toks,
                      "tps":round(m.get("tokens_per_second",0),1),
                      "peak_mb":round(m.get("peak_memory_mb",0),1), "wer":w,
                      "wall_s":round(wall,2), "text":text[:200]})

    valid = [r for r in runs if "error" not in r]
    if not valid:
        return {"impl":"swift", "model":model_label, "audio":audio_label, "error":"all failed", "runs":runs}
    avg_t = sum(r["time_s"] for r in valid)/len(valid)
    avg_tps = sum(r["tps"] for r in valid)/len(valid)
    return {"impl":"swift", "model":model_label, "audio":audio_label,
            "avg_time_s":round(avg_t,4), "avg_tps":round(avg_tps,1),
            "avg_peak_mb":round(sum(r["peak_mb"] for r in valid)/len(valid),1),
            "runs":runs}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for model_label, model_path in MODELS:
        for audio_label, audio_path, ref in AUDIO_FILES:
            print(f"\n{'='*70}")
            print(f"  {model_label} | {audio_label}")
            print(f"{'='*70}")

            # Python
            try:
                py = bench_python(model_label, model_path, audio_label, audio_path, ref)
                all_results.append(py)
                valid = [r for r in py["runs"] if "error" not in r]
                for i,r in enumerate(valid):
                    print(f"  [py]  run {i+1}: {r['time_s']:.3f}s | {r['tokens']} tok | {r['tps']:.0f} tok/s | mem {r['peak_mb']:.0f}MB" +
                          (f" | WER {r['wer']:.0%}" if r.get('wer') is not None else ""))
            except Exception as e:
                print(f"  [py]  ERROR: {e}")
                all_results.append({"impl":"python","model":model_label,"audio":audio_label,"error":str(e)})

            # Swift
            try:
                sw = bench_swift(model_label, model_path, audio_label, audio_path, ref)
                all_results.append(sw)
                valid = [r for r in sw.get("runs",[]) if "error" not in r]
                for i,r in enumerate(valid):
                    print(f"  [sw]  run {i+1}: {r['time_s']:.3f}s | {r['tokens']} tok | {r['tps']:.0f} tok/s | mem {r['peak_mb']:.0f}MB" +
                          (f" | WER {r['wer']:.0%}" if r.get('wer') is not None else "") +
                          f" (wall {r.get('wall_s',0):.1f}s)")
            except Exception as e:
                print(f"  [sw]  ERROR: {e}")
                all_results.append({"impl":"swift","model":model_label,"audio":audio_label,"error":str(e)})

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'Impl':<8} {'Model':<6} {'Audio':<22} {'Time':>8} {'Tok/s':>7} {'Mem MB':>7} {'Tokens':>7}")
    print("-"*90)
    for r in all_results:
        if "error" in r:
            print(f"{r['impl']:<8} {r['model']:<6} {r['audio']:<22} {'ERROR':>8}")
            continue
        print(f"{r['impl']:<8} {r['model']:<6} {r['audio']:<22} "
              f"{r['avg_time_s']:>7.3f}s {r['avg_tps']:>6.0f} {r['avg_peak_mb']:>6.0f} "
              f"{r['runs'][0]['tokens']:>7}")

    out_path = os.path.join(OUTPUT_DIR, "benchmark_full.json")
    with open(out_path, "w") as f:
        json.dump({"benchmark":"moonshine-mlx-full", "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "num_warmup":NUM_WARMUP, "num_runs":NUM_RUNS, "results":all_results}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
