# 🩺 Advanced AI Medical Intelligence Platform

An end-to-end AI system that analyzes chest X-ray images, predicts disease
(Pneumonia vs Normal) using a deep learning model, **explains** its
predictions visually with Grad-CAM, and generates an **LLM-assisted draft
report** (Gemini API) — all served through a REST API and a Streamlit UI,
with full prediction history stored in a database.

> ⚠️ **Disclaimer:** This is an educational/portfolio project. It is **not**
> a certified medical device and must never be used for real clinical
> decision-making. All AI-generated reports explicitly state this.

---

## ✨ Features

| Requirement | Implementation |
|---|---|
| Medical image analysis | Chest X-ray classification |
| Deep Learning disease prediction | ResNet18 (transfer learning, PyTorch) |
| Explainable AI | Grad-CAM heatmap overlay |
| LLM-assisted report generation | Google Gemini API (`google-genai`) |
| REST APIs | FastAPI (`/predict`, `/history`, ...) |
| Prediction history storage | SQLite via SQLAlchemy (swappable to Postgres) |
| User interface | Streamlit web app |
| Deployment | Dockerfile + docker-compose |

---

## 🏗️ Project Structure

```
medical-ai-platform/
├── model/
│   ├── model_def.py       # ResNet18 architecture
│   ├── train.py           # Training script (real or synthetic data)
│   ├── gradcam.py          # Grad-CAM explainability
│   └── dataset_prep.py     # Real-data instructions + synthetic data generator
├── backend/
│   ├── main.py             # FastAPI app & routes
│   ├── database.py         # SQLAlchemy models
│   ├── schemas.py          # Pydantic response models
│   ├── inference.py        # Model loading + prediction + Grad-CAM pipeline
│   ├── llm_report.py        # Gemini-based report generation
│   └── config.py            # Env-driven configuration
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/
│   └── test_api.py          # API smoke tests (pytest)
├── saved_models/             # Trained .pth checkpoints go here
├── gradcam_outputs/           # Saved Grad-CAM heatmap images
├── data/                       # Dataset lives here (gitignored)
├── docs/
│   └── report_template.md      # Source for the PDF project report
├── requirements.txt
├── Dockerfile                   # Backend image
├── Dockerfile.frontend          # Frontend image
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (local, no Docker)

### 1. Environment setup (Windows / Anaconda PowerShell)

```powershell
conda create -n medical-ai python=3.11 -y
conda activate medical-ai
pip install -r requirements.txt
copy .env.example .env
# edit .env and add your GEMINI_API_KEY
```

### 2. Get training data

**Option A — Real dataset (recommended for your actual submission):**
Download ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
from Kaggle and unzip so you have:

```
data/chest_xray/train/NORMAL, train/PNEUMONIA
data/chest_xray/val/NORMAL,   val/PNEUMONIA
data/chest_xray/test/NORMAL,  test/PNEUMONIA
```

**Option B — Synthetic demo data (instant, to verify the whole pipeline works):**

```powershell
cd model
python dataset_prep.py --synthetic
```

### 3. Train the model

```powershell
cd model
python train.py --data_dir ../data/chest_xray --epochs 10 --batch_size 32
```

This saves `saved_models/pneumonia_resnet18.pth` plus training history and
test metrics JSON files.

### 4. Run the backend API

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive Swagger API docs.

### 5. Run the frontend

In a second terminal:

```powershell
cd frontend
streamlit run app.py
```

Visit **http://localhost:8501**, upload an X-ray, click "Run AI Analysis".

---

## 🐳 Run everything with Docker Compose

```powershell
copy .env.example .env
# edit .env: set GEMINI_API_KEY
docker compose up --build
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

> Train the model **before** building the Docker image (or mount
> `saved_models/` as shown in `docker-compose.yml`) so the container has
> `pneumonia_resnet18.pth` available.

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Upload image → prediction + Grad-CAM + LLM report |
| GET | `/history?skip=0&limit=20` | Paginated prediction history |
| GET | `/history/{id}` | Full detail of one prediction |
| DELETE | `/history/{id}` | Delete a record |
| GET | `/gradcam/{filename}` | Serve a saved Grad-CAM image |

Example with `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_xray.png"
```

---

## 🧪 Testing

```powershell
cd backend
pytest ../tests -v
```

---

## 🧠 Model Details

- **Architecture:** ResNet18 (ImageNet-pretrained backbone, fine-tuned FC head)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Input size:** 224×224 RGB
- **Explainability:** Grad-CAM on `layer4[-1]` (last conv block)
- **Metrics tracked:** accuracy, precision, recall, F1, confusion matrix
  (saved to `saved_models/metrics.json` after training)

To extend to multi-class disease detection (e.g. COVID/Bacterial/Viral/Normal),
change `NUM_CLASSES` and `CLASS_NAMES` in `model/model_def.py` and structure
your dataset folders accordingly — no other code changes needed.

---

## 🔐 Environment Variables

See `.env.example`. Key ones:

- `GEMINI_API_KEY` — required for real LLM report generation (falls back to
  a templated disclaimer-only report if unset, so the app never crashes)
- `DATABASE_URL` — defaults to local SQLite, swappable to Postgres
- `MODEL_PATH`, `GRADCAM_DIR` — storage locations

---

## 📌 Roadmap / Possible Extensions

- Multi-class disease classification (COVID-19, TB, etc.)
- Postgres + Alembic migrations for production DB
- User authentication (JWT) for multi-user history
- Model versioning / MLflow experiment tracking
- CI pipeline (GitHub Actions) running `pytest` on push

---

## 📄 License

Educational/portfolio use.
