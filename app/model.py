import asyncio
import io

import magic
import numpy as np
import onnxruntime
from PIL import Image, ImageOps

# Module-level session variable — imported by endpoints
session: onnxruntime.InferenceSession | None = None

LABELS = ["artificial", "real"]
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = 256


def load_model(model_path: str = "models/model.onnx") -> onnxruntime.InferenceSession:
    """Load the ONNX model and return the InferenceSession."""
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return onnxruntime.InferenceSession(model_path, sess_options)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Validate, preprocess and return image as [1, 3, 256, 256] float32 array."""
    mime = magic.from_buffer(file_bytes, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported MIME type: {mime}. Allowed: {ALLOWED_MIME_TYPES}"
        )

    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0  # [256, 256, 3], [0,1]
    img_array = (img_array - MEAN) / STD  # ImageNet normalize
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW [3, 256, 256]
    img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 256, 256]
    return img_array.astype(np.float32)


async def predict_image(file_bytes: bytes) -> dict:
    """Run async inference and return label, confidence, and per-class probabilities."""
    input_array = preprocess_image(file_bytes)

    def sync_infer(arr: np.ndarray) -> np.ndarray:
        assert session is not None, "Model session is not loaded"
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: arr})
        return np.array(outputs[0])

    loop = asyncio.get_running_loop()
    logits = await loop.run_in_executor(None, sync_infer, input_array)

    # Softmax over logits (shape: [1, 2])
    logits = logits[0]  # [2]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    best_idx = int(np.argmax(probabilities))
    label = LABELS[best_idx]
    confidence = float(probabilities[best_idx])
    prob_dict = {LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))}

    return {"label": label, "confidence": confidence, "probabilities": prob_dict}
