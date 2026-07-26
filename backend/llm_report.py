"""
llm_report.py

Uses Groq's API (fast, free-tier friendly, no billing verification required)
to turn a raw model prediction into a readable, structured, patient/clinician
-friendly draft report.

IMPORTANT: this is an AI-generated *draft* to assist a radiologist/clinician
-- it is NOT a diagnosis and every report explicitly says so. Set your key
via the GROQ_API_KEY environment variable (see .env.example).
"""

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

REPORT_PROMPT = """You are assisting a radiologist by drafting a preliminary,
plain-language explanation of an AI model's chest X-ray screening output.
Do NOT present this as a final diagnosis. Always recommend confirmation by
a qualified radiologist/physician.

Model output:
- Predicted class: {predicted_class}
- Confidence: {confidence:.1%}
- Explainability note: Grad-CAM highlighted regions of the lung fields that
  most influenced this prediction (see attached heatmap).

Write a short structured report with these sections:
1. Summary (2-3 sentences, plain language)
2. Observations (what the class + confidence typically indicate, described
   generally, not as certainty)
3. Recommended Next Steps
4. Disclaimer (that this is an AI-assisted draft, not a medical diagnosis)

Keep it under 200 words total, clear and professional.
"""


def generate_report(predicted_class: str, confidence: float) -> str:
    if _client is None:
        return (
            "[LLM report unavailable: GROQ_API_KEY not set]\n\n"
            f"Model predicted: {predicted_class} (confidence {confidence:.1%}).\n"
            "This is an AI-assisted screening result only, not a medical diagnosis. "
            "Please consult a licensed radiologist/physician for confirmation."
        )

    prompt = REPORT_PROMPT.format(predicted_class=predicted_class, confidence=confidence)

    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        return (
            f"[LLM report generation failed: {e}]\n\n"
            f"Model predicted: {predicted_class} (confidence {confidence:.1%}). "
            "Please consult a licensed radiologist/physician for confirmation."
        )