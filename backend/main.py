"""
main.py

FastAPI backend for the Advanced AI Medical Intelligence Platform.

Endpoints:
    POST   /predict           -> upload an X-ray image, get prediction + Grad-CAM + LLM report
    GET    /history            -> list all past predictions (paginated)
    GET    /history/{id}       -> get one prediction record in detail
    DELETE /history/{id}       -> delete a record
    GET    /gradcam/{filename} -> serve a saved Grad-CAM heatmap image
    GET    /health              -> health check

Run:
    uvicorn main:app --reload --port 8000
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from config import GRADCAM_DIR
from database import init_db, get_db, PredictionRecord
from schemas import PredictionResponse, HistoryItem
from inference import predict_and_explain
from llm_report import generate_report

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="Deep learning disease prediction + Grad-CAM explainability + LLM-assisted reporting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.on_event("startup")
def on_startup():
    init_db()


# Also initialize immediately on import (covers test runners / ASGI setups
# that don't emit the startup event, e.g. a plain TestClient() without a
# context manager).
init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    result = predict_and_explain(image_bytes, file.filename)
    report_text = generate_report(result["predicted_class"], result["confidence"])

    record = PredictionRecord(
        filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        gradcam_path=result["gradcam_filename"],
        llm_report=report_text,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        gradcam_url=f"/gradcam/{record.gradcam_path}",
        llm_report=record.llm_report,
        created_at=record.created_at,
    )


@app.get("/history", response_model=list[HistoryItem])
def get_history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return records


@app.get("/history/{record_id}", response_model=PredictionResponse)
def get_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        gradcam_url=f"/gradcam/{record.gradcam_path}",
        llm_report=record.llm_report,
        created_at=record.created_at,
    )


@app.delete("/history/{record_id}")
def delete_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"deleted": record_id}


@app.get("/gradcam/{filename}")
def get_gradcam_image(filename: str):
    path = os.path.join(GRADCAM_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")
