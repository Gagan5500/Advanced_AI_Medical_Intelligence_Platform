"""
frontend/app.py

Streamlit UI for the Advanced AI Medical Intelligence Platform.

Run:
    streamlit run app.py

Set BACKEND_URL env var if the FastAPI backend isn't on localhost:8000
(e.g. inside Docker Compose it will be http://backend:8000).

Set PUBLIC_BACKEND_URL for URLs the BROWSER needs to fetch directly
(like Grad-CAM images) — inside Docker this must be a host-reachable
address like http://localhost:8000, NOT the internal service name.
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Medical Intelligence Platform", page_icon="🩺", layout="wide")

st.title("🩺 Advanced AI Medical Intelligence Platform")
st.caption(
    "Deep learning chest X-ray screening with Grad-CAM explainability and "
    "LLM-assisted draft reporting. **Not a substitute for professional medical diagnosis.**"
)

tab_predict, tab_history = st.tabs(["🔍 New Prediction", "📜 History"])

with tab_predict:
    uploaded = st.file_uploader("Upload a chest X-ray image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    if uploaded is not None:
        with col1:
            st.image(uploaded, caption="Uploaded X-ray", use_column_width=True)

        if st.button("Run AI Analysis", type="primary"):
            with st.spinner("Running model inference + Grad-CAM + LLM report..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()

                    with col2:
                        gradcam_url = f"{PUBLIC_BACKEND_URL}{data['gradcam_url']}"
                        st.image(gradcam_url, caption="Grad-CAM Explanation", use_column_width=True)

                    st.divider()
                    label = data["predicted_class"]
                    conf = data["confidence"]

                    badge_color = "red" if label != "NORMAL" else "green"
                    st.markdown(f"### Prediction: :{badge_color}[{label}]")
                    st.progress(min(conf, 1.0), text=f"Confidence: {conf:.1%}")

                    st.subheader("📄 AI-Assisted Draft Report")
                    st.info(data["llm_report"])

                    st.caption(
                        "⚠️ This is an AI-generated screening aid, not a medical diagnosis. "
                        "Always consult a licensed radiologist or physician."
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")

with tab_history:
    st.subheader("Prediction History")
    limit = st.slider("Records to show", 5, 100, 20)

    try:
        resp = requests.get(f"{BACKEND_URL}/history", params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            st.write("No predictions yet — run one from the 'New Prediction' tab.")
        else:
            for r in records:
                with st.expander(f"#{r['id']} — {r['filename']} — {r['predicted_class']} ({r['confidence']:.1%})"):
                    detail_resp = requests.get(f"{BACKEND_URL}/history/{r['id']}", timeout=30)
                    if detail_resp.ok:
                        detail = detail_resp.json()
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.image(f"{PUBLIC_BACKEND_URL}{detail['gradcam_url']}", use_column_width=True)
                        with c2:
                            st.write(detail["llm_report"])
                        if st.button("Delete", key=f"del_{r['id']}"):
                            requests.delete(f"{BACKEND_URL}/history/{r['id']}", timeout=30)
                            st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend at {BACKEND_URL}: {e}")