"""
Export Modotte/AIRealNet to INT8 ONNX using Optimum + onnxruntime quantization.

Usage:
    python scripts/export_onnx.py

Output:
    models/model.onnx  (INT8 quantized, QUInt8)
"""

# NOTE: Set HF_ENDPOINT env var to use a mirror (e.g. https://hf-mirror.com)

import shutil
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForImageClassification

MODEL_ID = "Modotte/AIRealNet"
MODELS_DIR = Path("models")
ONNX_EXPORT_DIR = MODELS_DIR / "_onnx_export_tmp"
FINAL_MODEL = MODELS_DIR / "model.onnx"
QUANTIZED_MODEL = MODELS_DIR / "_quantized_tmp.onnx"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Exporting '{MODEL_ID}' → ONNX (fp32) via Optimum ...")
    model = ORTModelForImageClassification.from_pretrained(
        MODEL_ID,
        export=True,
    )
    model.save_pretrained(str(ONNX_EXPORT_DIR))
    print(f"      Saved fp32 ONNX to: {ONNX_EXPORT_DIR}")

    for f in ONNX_EXPORT_DIR.rglob("*.onnx"):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"      • {f.name}  ({size_mb:.1f} MB)")


    candidates = list(ONNX_EXPORT_DIR.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx file found in {ONNX_EXPORT_DIR}")
    fp32_model_path = candidates[0]

    print("[2/4] Applying INT8 dynamic quantization (QUInt8, generic x86) ...")
    quantize_dynamic(
        model_input=str(fp32_model_path),
        model_output=str(QUANTIZED_MODEL),
        weight_type=QuantType.QUInt8,
    )
    quant_size_mb = QUANTIZED_MODEL.stat().st_size / 1024 / 1024
    print(f"      Quantized model: {QUANTIZED_MODEL}  ({quant_size_mb:.1f} MB)")

    print(f"[3/4] Moving quantized model → {FINAL_MODEL} ...")
    shutil.move(str(QUANTIZED_MODEL), str(FINAL_MODEL))
    final_size_mb = FINAL_MODEL.stat().st_size / 1024 / 1024
    print(f"      ✓ {FINAL_MODEL}  ({final_size_mb:.1f} MB)")

    if final_size_mb > 100:
        print(f"  ⚠  WARNING: model is {final_size_mb:.1f} MB — exceeds 100 MB target!")
    else:
        print(f"  ✓  Model size OK: {final_size_mb:.1f} MB < 100 MB")

    print("[4/4] Running smoke test ...")
    sess = onnxruntime.InferenceSession(str(FINAL_MODEL))
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    outputs = sess.run(None, {input_name: dummy})
    assert outputs[0].shape == (1, 2), (
        f"Smoke test FAILED: expected output shape (1, 2), got {outputs[0].shape}"
    )
    print(f"      ✓ Output shape: {outputs[0].shape} — smoke test passed!")


    shutil.rmtree(ONNX_EXPORT_DIR, ignore_errors=True)
    print("      Cleaned up intermediate export dirs.")
    print("\nDone!")


if __name__ == "__main__":
    main()
