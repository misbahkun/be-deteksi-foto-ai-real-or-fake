import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

VALID_LABELS = {"artificial", "real"}


def _make_image_bytes(fmt: str, width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(128, 64, 200)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient):
        assert client.get("/health").status_code == 200

    def test_model_loaded_is_true(self, client: TestClient):
        assert client.get("/health").json()["model_loaded"] is True

    def test_status_ok(self, client: TestClient):
        assert client.get("/health").json()["status"] == "ok"


class TestPredictHappyPath:
    def test_jpeg_returns_200(self, client: TestClient):
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_response_label_is_valid(self, client: TestClient):
        body = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        ).json()
        assert body["label"] in VALID_LABELS

    def test_response_confidence_in_range(self, client: TestClient):
        body = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        ).json()
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_probabilities_keys(self, client: TestClient):
        probs = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        ).json()["probabilities"]
        assert set(probs.keys()) == VALID_LABELS

    def test_probabilities_sum_to_one(self, client: TestClient):
        probs = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        ).json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_inference_time_ms_is_non_negative_int(self, client: TestClient):
        body = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        ).json()
        assert isinstance(body["inference_time_ms"], int)
        assert body["inference_time_ms"] >= 0

    def test_png_accepted(self, client: TestClient):
        resp = client.post(
            "/predict",
            files={"file": ("test.png", _make_image_bytes("PNG"), "image/png")},
        )
        assert resp.status_code == 200

    def test_webp_accepted(self, client: TestClient):
        resp = client.post(
            "/predict",
            files={"file": ("test.webp", _make_image_bytes("WEBP"), "image/webp")},
        )
        assert resp.status_code == 200


class TestPredictErrorCases:
    def test_text_file_returns_400(self, client: TestClient):
        text_bytes = b"This is not an image file at all.\n" * 10
        resp = client.post(
            "/predict",
            files={"file": ("notes.txt", text_bytes, "text/plain")},
        )
        assert resp.status_code == 400

    def test_binary_non_image_returns_400(self, client: TestClient):
        garbage = bytes(range(256)) * 16
        resp = client.post(
            "/predict",
            files={"file": ("garbage.bin", garbage, "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self, client: TestClient):
        big_bytes = b"\xff" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            "/predict",
            files={"file": ("big.jpg", big_bytes, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_oversized_detail_mentions_10mb(self, client: TestClient):
        big_bytes = b"\xff" * (10 * 1024 * 1024 + 1)
        body = client.post(
            "/predict",
            files={"file": ("big.jpg", big_bytes, "image/jpeg")},
        ).json()
        assert "detail" in body
        assert "10MB" in body["detail"] or "10" in body["detail"]

    def test_exactly_10mb_not_413(self, client: TestClient):
        exactly_10mb = b"\xff" * (10 * 1024 * 1024)
        resp = client.post(
            "/predict",
            files={"file": ("exact.jpg", exactly_10mb, "image/jpeg")},
        )
        assert resp.status_code != 413
