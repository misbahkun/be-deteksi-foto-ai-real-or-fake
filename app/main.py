import io
import os
import time
from contextlib import asynccontextmanager

import google.generativeai as genai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import app.model as model_module
from app.schemas import PredictResponse

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def generate_gemini_message(label: str, confidence: float, image_bytes: bytes) -> str:
    """Generate a friendly Bahasa Indonesia comment via Gemini API using vision."""
    default_message = (
        f"Gambar ini terdeteksi sebagai {label} "
        f"dengan tingkat kepercayaan {confidence:.2f}."
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return default_message

    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-3-flash-preview")

        image = Image.open(io.BytesIO(image_bytes))

        prompt = (
            f"Halo! Kamu adalah PindAI, asisten virtual pintar pendeteksi gambar AI. "
            f"Sistem core ONNX kami mendeteksi gambar yang dilampirkan ini sebagai '{label.upper()}' "
            f"dengan tingkat keyakinan {confidence*100:.1f}%. "
            f"Tugasmu: Analisis gambar ini secara singkat, lalu berikan penjelasan visual yang mendukung "
            f"kenapa gambar ini terlihat asli/buatan AI. "
            f"Gunakan gaya bahasa santai, asik, tapi tetap sopan (seperti teman yang pintar). "
            f"Cukup 2-3 kalimat saja."
        )

        response = gemini_model.generate_content([prompt, image])
        return response.text.strip()
    except Exception:
        return default_message


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model on startup, release on shutdown."""
    model_module.session = model_module.load_model()
    yield
    model_module.session = None


app = FastAPI(title="AI Image Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_module.session is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if len(file_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB.",
        )

    start = time.perf_counter()

    try:
        result = await model_module.predict_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    inference_time_ms = int((time.perf_counter() - start) * 1000)

    message = await generate_gemini_message(result["label"], result["confidence"], file_bytes)

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_time_ms=inference_time_ms,
        message=message,
    )
