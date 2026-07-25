import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "saved_models", "pneumonia_resnet18.pth"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'predictions.db')}")
GRADCAM_DIR = os.getenv("GRADCAM_DIR", os.path.join(BASE_DIR, "gradcam_outputs"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
