from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8,
                           textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6,
                           textColor=colors.HexColor("#2c5f8a")))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["Normal"], fontSize=10.5, leading=15,
                           spaceAfter=8))
styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontSize=26, leading=32,
                           textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=13, leading=18,
                           alignment=1, textColor=colors.HexColor("#444444")))

story = []

# ---------- Cover ----------
story.append(Spacer(1, 2 * inch))
story.append(Paragraph("Advanced AI Medical Intelligence Platform", styles["Cover"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "Project Report<br/>Deep Learning &bull; Explainable AI &bull; LLM-Assisted Reporting &bull; REST API &bull; Deployment",
    styles["CoverSub"]))
story.append(Spacer(1, 1.5 * inch))
story.append(Paragraph("Prepared as a technical portfolio project.", styles["CoverSub"]))
story.append(PageBreak())

# ---------- 1. Objective ----------
story.append(Paragraph("1. Project Objective", styles["H1Custom"]))
story.append(Paragraph(
    "This project implements a complete, end-to-end AI application for medical image "
    "analysis. It classifies chest X-ray images as NORMAL or PNEUMONIA using a deep "
    "convolutional neural network, explains each prediction visually using Grad-CAM "
    "(Gradient-weighted Class Activation Mapping), generates a plain-language draft "
    "report using a Large Language Model (Google Gemini), exposes all functionality "
    "through a documented REST API, persists every prediction in a relational database, "
    "and ships with a Streamlit-based web interface plus Docker-based deployment.",
    styles["BodyCustom"]))
story.append(Paragraph(
    "<b>Disclaimer:</b> This system is an educational/portfolio project. It is not a "
    "certified medical device and is not intended for real clinical diagnosis. Every "
    "AI-generated report explicitly states this limitation.",
    styles["BodyCustom"]))

# ---------- 2. System Architecture ----------
story.append(Paragraph("2. System Architecture", styles["H1Custom"]))
story.append(Paragraph(
    "The system follows a modular, three-tier architecture:", styles["BodyCustom"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Model layer</b> (model/): ResNet18 CNN, training pipeline, "
                        "Grad-CAM explainability module, dataset preparation utilities.",
                        styles["BodyCustom"])),
    ListItem(Paragraph("<b>Backend layer</b> (backend/): FastAPI REST API, SQLAlchemy ORM "
                        "for database persistence, Gemini-based LLM report generator, "
                        "inference pipeline tying the model and Grad-CAM together.",
                        styles["BodyCustom"])),
    ListItem(Paragraph("<b>Frontend layer</b> (frontend/): Streamlit web application for "
                        "image upload, result visualization, and history browsing.",
                        styles["BodyCustom"])),
], bulletType="bullet"))
story.append(Paragraph(
    "Both backend and frontend are independently containerized and orchestrated together "
    "via Docker Compose, so the entire platform can be brought up with a single command.",
    styles["BodyCustom"]))

# ---------- 3. Dataset ----------
story.append(Paragraph("3. Dataset", styles["H1Custom"]))
story.append(Paragraph(
    "The model is designed for the public 'Chest X-Ray Images (Pneumonia)' dataset "
    "(Kermany et al.), containing labeled NORMAL and PNEUMONIA chest radiographs split "
    "into train/validation/test sets. A synthetic data generator "
    "(model/dataset_prep.py --synthetic) is also included, producing procedurally "
    "generated X-ray-like images so the entire pipeline (training, inference, Grad-CAM, "
    "API, database, UI) can be verified end-to-end without requiring the ~2GB download, "
    "which is useful for CI/demo purposes.",
    styles["BodyCustom"]))

# ---------- 4. Deep Learning Model ----------
story.append(Paragraph("4. Deep Learning Model", styles["H1Custom"]))
data = [
    ["Component", "Detail"],
    ["Architecture", "ResNet18 (ImageNet-pretrained backbone, fine-tuned FC head)"],
    ["Framework", "PyTorch / TorchVision"],
    ["Input", "224 x 224 RGB image"],
    ["Output classes", "NORMAL, PNEUMONIA (binary; extensible to multi-class)"],
    ["Loss function", "Cross-Entropy Loss"],
    ["Optimizer", "Adam (lr=1e-4) with StepLR scheduler"],
    ["Augmentation", "Random horizontal flip, random rotation (+/-10 deg)"],
    ["Evaluation metrics", "Accuracy, Precision, Recall, F1-score, Confusion Matrix"],
]
t = Table(data, colWidths=[1.8 * inch, 4.3 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "Training and test metrics are automatically written to "
    "saved_models/training_history.json and saved_models/metrics.json after each run, "
    "so performance is fully reproducible and auditable.", styles["BodyCustom"]))

# ---------- 5. Explainable AI ----------
story.append(Paragraph("5. Explainable AI (Grad-CAM)", styles["H1Custom"]))
story.append(Paragraph(
    "To avoid a 'black box' prediction, every inference request also runs Grad-CAM on "
    "the final convolutional block (layer4) of the ResNet18 backbone. Grad-CAM computes "
    "gradients of the predicted class score with respect to the feature maps of that "
    "layer, weights the activations accordingly, and produces a heatmap highlighting the "
    "image regions most responsible for the prediction. This heatmap is overlaid on the "
    "original X-ray and returned alongside every prediction &mdash; giving a visual, "
    "clinically-interpretable explanation rather than a bare probability score.",
    styles["BodyCustom"]))

# ---------- 6. LLM Integration ----------
story.append(Paragraph("6. LLM-Assisted Report Generation", styles["H1Custom"]))
story.append(Paragraph(
    "The backend integrates Google's Gemini API (via the google-genai SDK) to translate "
    "the raw model output (predicted class + confidence) into a short, structured, "
    "plain-language draft report with four sections: Summary, Observations, Recommended "
    "Next Steps, and a mandatory Disclaimer clarifying that the output is an AI-assisted "
    "screening aid, not a diagnosis. If no API key is configured, the system gracefully "
    "falls back to a templated disclaimer-only report rather than failing, so the "
    "application remains usable end-to-end in any environment.",
    styles["BodyCustom"]))

# ---------- 7. REST API ----------
story.append(Paragraph("7. REST API Design", styles["H1Custom"]))
api_data = [
    ["Method", "Endpoint", "Purpose"],
    ["GET", "/health", "Health check"],
    ["POST", "/predict", "Upload image -> prediction + Grad-CAM + LLM report"],
    ["GET", "/history", "Paginated list of past predictions"],
    ["GET", "/history/{id}", "Full detail of one prediction"],
    ["DELETE", "/history/{id}", "Delete a prediction record"],
    ["GET", "/gradcam/{filename}", "Serve a saved Grad-CAM heatmap image"],
]
t2 = Table(api_data, colWidths=[0.9 * inch, 1.7 * inch, 3.5 * inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t2)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "Built with FastAPI, the API is fully documented via auto-generated OpenAPI/Swagger "
    "docs at /docs, uses Pydantic for request/response validation, and includes CORS "
    "middleware for frontend integration.", styles["BodyCustom"]))

# ---------- 8. Database ----------
story.append(Paragraph("8. Database Design", styles["H1Custom"]))
story.append(Paragraph(
    "Prediction history is persisted using SQLAlchemy ORM against a SQLite database by "
    "default (zero-config), with a single DATABASE_URL environment variable swap needed "
    "to move to PostgreSQL or MySQL in production. Each record stores: id, filename, "
    "predicted class, confidence score, path to the saved Grad-CAM heatmap image, the "
    "generated LLM report text, and a UTC timestamp.", styles["BodyCustom"]))

# ---------- 9. Web Application ----------
story.append(Paragraph("9. Web Application (Streamlit)", styles["H1Custom"]))
story.append(Paragraph(
    "The frontend provides two views: (1) a New Prediction tab for uploading an X-ray, "
    "triggering analysis, and viewing the prediction, confidence, Grad-CAM overlay, and "
    "LLM report side-by-side; and (2) a History tab listing all past predictions with "
    "expandable detail (heatmap + report) and delete functionality, fetched live from the "
    "REST API.", styles["BodyCustom"]))

# ---------- 10. Deployment ----------
story.append(Paragraph("10. Deployment", styles["H1Custom"]))
story.append(Paragraph(
    "The project includes a Dockerfile for the FastAPI backend and a separate "
    "Dockerfile.frontend for the Streamlit app, orchestrated together with "
    "docker-compose.yml. A single 'docker compose up --build' command builds and starts "
    "both services, with the trained model checkpoint, Grad-CAM output directory, and "
    "SQLite database file mounted as persistent volumes.", styles["BodyCustom"]))

# ---------- 11. Testing ----------
story.append(Paragraph("11. Testing", styles["H1Custom"]))
story.append(Paragraph(
    "A pytest-based smoke test suite (tests/test_api.py) uses FastAPI's TestClient to "
    "verify the health check, the full predict -> history -> detail -> delete workflow, "
    "and input validation (rejecting non-image uploads). This suite was executed during "
    "development and passes end-to-end, confirming the model, Grad-CAM, database, and "
    "LLM fallback path all integrate correctly.", styles["BodyCustom"]))

# ---------- 12. Best Practices ----------
story.append(Paragraph("12. Best Practices Followed", styles["H1Custom"]))
story.append(ListFlowable([
    ListItem(Paragraph("Modular, separation-of-concerns project structure (model / backend / frontend)", styles["BodyCustom"])),
    ListItem(Paragraph("Environment-variable-driven configuration (.env.example) &mdash; no hardcoded secrets", styles["BodyCustom"])),
    ListItem(Paragraph("Graceful degradation: app remains functional without a Gemini API key or GPU", styles["BodyCustom"])),
    ListItem(Paragraph("Input validation and structured error handling on all API endpoints", styles["BodyCustom"])),
    ListItem(Paragraph("Automated tests and reproducible training metrics", styles["BodyCustom"])),
    ListItem(Paragraph("Containerized, one-command deployment via Docker Compose", styles["BodyCustom"])),
], bulletType="bullet"))

# ---------- 13. Future Work ----------
story.append(Paragraph("13. Future Work", styles["H1Custom"]))
story.append(ListFlowable([
    ListItem(Paragraph("Multi-class disease classification (COVID-19, tuberculosis, etc.)", styles["BodyCustom"])),
    ListItem(Paragraph("User authentication (JWT) for multi-user prediction history", styles["BodyCustom"])),
    ListItem(Paragraph("PostgreSQL + Alembic migrations for production-scale deployment", styles["BodyCustom"])),
    ListItem(Paragraph("CI/CD pipeline (GitHub Actions) running the test suite on every push", styles["BodyCustom"])),
    ListItem(Paragraph("Model experiment tracking with MLflow or Weights & Biases", styles["BodyCustom"])),
], bulletType="bullet"))

doc = SimpleDocTemplate("/home/claude/medical-ai-platform/docs/Project_Report.pdf",
                         pagesize=letter,
                         topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                         leftMargin=0.9 * inch, rightMargin=0.9 * inch)
doc.build(story)
print("PDF generated.")
