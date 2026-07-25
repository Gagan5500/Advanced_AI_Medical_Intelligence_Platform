"""
tests/test_api.py

Basic smoke tests for the FastAPI backend. Run with:
    cd backend && pytest ../tests -v

Uses FastAPI's TestClient, so it spins the app in-process (no server needed).
Note: /predict test creates a tiny synthetic image on the fly, so it works
even before a real model is trained (though prediction quality is untrained).
"""

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def _fake_image_bytes():
    img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_and_history_flow():
    img_buf = _fake_image_bytes()
    resp = client.post("/predict", files={"file": ("test.png", img_buf, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "gradcam_url" in data
    assert "llm_report" in data

    record_id = data["id"]

    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    assert any(r["id"] == record_id for r in hist_resp.json())

    detail_resp = client.get(f"/history/{record_id}")
    assert detail_resp.status_code == 200

    del_resp = client.delete(f"/history/{record_id}")
    assert del_resp.status_code == 200


def test_predict_rejects_bad_filetype():
    resp = client.post("/predict", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
    assert resp.status_code == 400
