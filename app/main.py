import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import app.model as model_module
from app.schemas import PredictResponse

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


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

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_time_ms=inference_time_ms,
    )
