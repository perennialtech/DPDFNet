"""Measure complete CPU streaming calls, not just neural-network execution."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpdfnet8_48khz_hr")
    parser.add_argument("--onnx-path", type=Path)
    parser.add_argument(
        "--backend", choices=["onnx", "openvino", "native"], default="onnx"
    )
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-weights", type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument(
        "--audio", type=Path, help="Optional recorded audio; resampled before timing"
    )
    parser.add_argument(
        "--cpu", type=int, help="Pin this benchmark process to a Linux CPU"
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        help="Submit a hop every 10 ms instead of running flat out",
    )
    parser.add_argument("--json", type=Path, help="Write the report to this file too")
    parser.add_argument(
        "--save-output", type=Path, help="Save enhanced samples as a NumPy array"
    )
    args = parser.parse_args()
    if args.backend == "native":
        if not args.native_library or not args.native_weights or args.onnx_path:
            parser.error("native needs --native-library and --native-weights, not --onnx-path")
    elif not args.onnx_path or args.native_library or args.native_weights:
        parser.error("onnx/openvino needs --onnx-path, not native paths")
    if args.seconds <= 0 or args.warmup_frames < 0:
        parser.error("--seconds must be positive and --warmup-frames non-negative")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    from dpdfnet import StreamEnhancer
    from dpdfnet.audio import ensure_sample_rate, to_mono
    from dpdfnet.models import get_model_info

    sr = get_model_info(args.model).sample_rate
    hop = sr // 100
    if args.audio:
        import soundfile as sf

        audio, input_sr = sf.read(args.audio, dtype="float32")
        audio = ensure_sample_rate(to_mono(audio), input_sr, sr)[
            : int(sr * args.seconds)
        ]
    else:
        # Deterministic non-speech workload with harmonics, noise and silence.
        rng = np.random.default_rng(2026)
        t = np.arange(int(sr * args.seconds)) / sr
        audio = (
            0.1 * np.sin(2 * np.pi * 230 * t) + 0.03 * rng.standard_normal(t.size)
        ).astype(np.float32)
        audio[(t % 4 > 2) & (t % 4 < 3)] = 0
    blocks = len(audio) // hop
    if blocks <= args.warmup_frames + 1:
        parser.error("Audio must contain more hops than --warmup-frames plus one")

    start = time.perf_counter_ns()
    kwargs = {"backend": args.backend} if args.backend != "onnx" else {}
    if args.backend == "native":
        kwargs.update(native_library=args.native_library, native_weights=args.native_weights)
    enhancer = StreamEnhancer(model=args.model, onnx_path=args.onnx_path, **kwargs)
    setup_ms = (time.perf_counter_ns() - start) / 1e6
    timings = np.empty(blocks, dtype=np.float64)
    lateness = np.zeros(blocks, dtype=np.float64)
    output = []
    first_inference_ms = None
    returned_samples = 0
    next_arrival = time.perf_counter()
    for index in range(blocks):
        chunk = audio[index * hop : (index + 1) * hop]
        if args.paced:
            time.sleep(max(0, next_arrival - time.perf_counter()))
            lateness[index] = max(0, time.perf_counter() - next_arrival) * 1000
            next_arrival += 0.010
        start = time.perf_counter_ns()
        enhanced = enhancer.process(chunk, sample_rate=sr)
        timings[index] = (time.perf_counter_ns() - start) / 1e6
        returned_samples += enhanced.size
        if enhanced.size and first_inference_ms is None:
            first_inference_ms = timings[index]
        if args.save_output:
            output.append(enhanced)
    if args.save_output:
        output.append(enhancer.flush())
        np.save(args.save_output, np.concatenate(output))

    measured = timings[args.warmup_frames :]
    packages = {"numpy": version("numpy")}
    if args.backend != "native":
        packages["onnxruntime"] = version("onnxruntime")
    if args.backend == "openvino":
        packages["openvino"] = version("openvino")
    rss_scale = 1024**2 if platform.system() == "Darwin" else 1024
    report = {
        "model": args.model,
        "model_sha256": hashlib.sha256(
            (
                args.native_weights if args.backend == "native" else args.onnx_path
            ).read_bytes()
        ).hexdigest(),
        "backend": args.backend,
        "paced": args.paced,
        "sample_rate": sr,
        "hop_samples": hop,
        "audio_source": (
            str(args.audio)
            if args.audio
            else "synthetic harmonics/noise/silence (not speech)"
        ),
        "audio_seconds": blocks / 100,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "setup_ms": setup_ms,
        "first_inference_call_ms": first_inference_ms,
        "warmup_hops_excluded": args.warmup_frames,
        "measured_hops": measured.size,
        "returned_samples_before_flush": returned_samples,
        "call_ms": dict(
            zip(
                ["p50", "p95", "p99", "max"], np.percentile(measured, [50, 95, 99, 100])
            )
        ),
        "mean_rtf": float(measured.mean() / 10),
        "calls_over_10ms": int(np.count_nonzero(measured > 10)),
        "completion_deadline_misses": int(
            np.count_nonzero(measured + lateness[args.warmup_frames :] > 10)
        ),
        "scheduler_lateness_p99_ms": float(
            np.percentile(lateness[args.warmup_frames :], 99)
        ),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / rss_scale,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
