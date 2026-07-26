import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "saved_models", "pneumonia_resnet18.pth"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'predictions.db')}")
GRADCAM_DIR = os.getenv("GRADCAM_DIR", os.path.join(BASE_DIR, "gradcam_outputs"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
