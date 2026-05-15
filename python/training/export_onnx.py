import torch
import numpy as np
import time
from pathlib import Path

import onnx
import onnxruntime as ort

from model import LODPerceptionNet

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "data/models"
ONNX_PATH = MODELS_DIR / "lod_perception.onnx"

# Export
def export(checkpoint_path: Path, onnx_path: Path) -> LODPerceptionNet:
    device     = torch.device("cpu")   # export from CPU for maximum compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint["config"]

    model = LODPerceptionNet(config["input_dim"], config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, config["input_dim"])

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,       # fold constant ops for smaller graph
        input_names=["features"],
        output_names=["quality_score"],
        dynamic_axes={
            "features":     {0: "batch_size"},
            "quality_score":{0: "batch_size"},
        },
    )
    print(f"  Exported: {onnx_path}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.1f} KB")
    return model


# Verify
def verify(model: LODPerceptionNet, onnx_path: Path):
    # Validate ONNX graph structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX graph structure: valid")

    # Numerical comparison
    session    = ort.InferenceSession(str(onnx_path),
                                      providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    np.random.seed(0)
    test_inputs = np.random.randn(10, 38).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.tensor(test_inputs)).numpy()

    onnx_out = session.run(None, {input_name: test_inputs})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"  Max absolute diff (PyTorch vs ONNX): {max_diff:.2e}  "
          f"({'PASS' if max_diff < 1e-5 else 'FAIL'})")
    return max_diff < 1e-5


# Benchmark
def benchmark(onnx_path: Path, n_warmup: int = 50, n_runs: int = 1000):
    providers = ["CPUExecutionProvider"]
    session   = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_name  = session.get_inputs()[0].name
    sample    = np.random.randn(1, 38).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run(None, {inp_name: sample})

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {inp_name: sample})
        times.append((time.perf_counter() - t0) * 1000)   # ms

    times = np.array(times)
    print(f"\n  CPU inference benchmark ({n_runs} runs, batch=1):")
    print(f"    Mean   : {times.mean():.4f} ms")
    print(f"    Median : {np.median(times):.4f} ms")
    print(f"    P95    : {np.percentile(times, 95):.4f} ms")
    print(f"    P99    : {np.percentile(times, 99):.4f} ms")
    print(f"    Min    : {times.min():.4f} ms")
    print(f"    Max    : {times.max():.4f} ms")
    print(f"    Target : <1.0000 ms  "
          f"({'PASS' if times.mean() < 1.0 else 'FAIL'})")

    # Batch benchmark
    batch    = np.random.randn(100, 38).astype(np.float32)
    t0       = time.perf_counter()
    for _ in range(100):
        session.run(None, {inp_name: batch})
    batch_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"\n  CPU batch inference (batch=100):")
    print(f"    Mean   : {batch_ms:.4f} ms per batch")
    print(f"    Per sample : {batch_ms/100:.4f} ms")

    return float(times.mean())


# MAIN
def main():
    print("=" * 55)
    print("  Acuity — ONNX Export")
    print("=" * 55)

    print("\n[1/3] Exporting to ONNX...")
    model = export(MODELS_DIR / "best_model.pth", ONNX_PATH)

    print("\n[2/3] Verifying numerical accuracy...")
    passed = verify(model, ONNX_PATH)

    print("\n[3/3] Benchmarking inference speed...")
    mean_ms = benchmark(ONNX_PATH)

    print("\n" + "=" * 55)
    print("  Export Summary")
    print("=" * 55)
    print(f"  ONNX path     : {ONNX_PATH}")
    print(f"  Verification  : {'PASS' if passed else 'FAIL'}")
    print(f"  Mean latency  : {mean_ms:.4f} ms  "
          f"({'PASS' if mean_ms < 1.0 else 'FAIL — exceeds 1ms target'})")
    print("=" * 55)


if __name__ == "__main__":
    main()