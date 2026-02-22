from pydantic import BaseModel


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: int
    message: str
