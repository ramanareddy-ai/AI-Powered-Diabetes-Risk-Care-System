"""
Diagnosis page for the Diabetes Risk & Care System.

This module defines the ``render()`` function that collects user inputs,
trains a logistic regression model (or loads a pre‑trained model), predicts
the probability of diabetes, and uses a language model to provide an
interpretive explanation.
"""
from __future__ import annotations

"""
Diagnosis page (humanized + visual + robust PDF).

Flow:
1) Load dataset safely (fallback to synthetic if missing).
2) Train a simple logistic regression.
3) Collect patient-entered metrics (+ patient name / case ID / logo).
4) Predict probability + label.
5) Show visual feedback in the app.
6) Generate a professional A4 PDF (Unicode, logo, header, charts) and auto-save.
"""

from typing import Dict, Tuple, List
from datetime import datetime
from pathlib import Path
import tempfile
import textwrap
import re
import os

import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF

from app.core.data import (
    load_diabetes_csv_or_synthesize,
    train_model,
    predict_risk,
)
from app.core.llm import chat  # uses OPENAI_API_KEY from secrets/env

# ---------- Paths / Assets ----------
ASSETS_DIR = Path("app/assets")
FONTS_DIR = ASSETS_DIR / "fonts"
DEFAULT_LOGO = ASSETS_DIR / "images" / "logo.png"  # optional fallback logo
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# ---------- PDF helpers ----------
def _pdf_init_a4_unicode() -> Tuple[FPDF, Tuple[str, str, int], Tuple[str, str, int], Tuple[str, str, int], bool]:
    """
    Create an A4 FPDF instance and register a Unicode font if available.
    Returns:
        (pdf, default_font, bold_font, header_font, unicode_ok)
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    font_path = FONTS_DIR / "DejaVuSans.ttf"
    try:
        if font_path.exists():
            pdf.add_font("DejaVu", "", str(font_path), uni=True)
            pdf.add_font("DejaVu", "B", str(font_path), uni=True)  # reuse for "bold"
            default_font = ("DejaVu", "", 11)
            bold_font = ("DejaVu", "B", 14)
            header_font = ("DejaVu", "B", 18)
            return pdf, default_font, bold_font, header_font, True
    except Exception:
        # Fallback gracefully if font registration fails
        pass

    # Latin-1 fallback
    default_font = ("Arial", "", 11)
    bold_font = ("Arial", "B", 14)
    header_font = ("Arial", "B", 18)
    return pdf, default_font, bold_font, header_font, False


def _sanitize_if_needed(s: str, unicode_ok: bool) -> str:
    """If not using a Unicode font, strip characters core fonts can’t render."""
    return s if unicode_ok else s.encode("latin-1", "ignore").decode("latin-1")


def _force_wrap_tokens(s: str, max_chars: int = 25) -> str:
    """
    Insert spaces inside any very long token to guarantee wrap.
    Prevents FPDF 'not enough horizontal space' errors.
    """
    def breaker(m):
        tok = m.group(0)
        return " ".join(tok[i:i + max_chars] for i in range(0, len(tok), max_chars))
    return re.sub(rf"\S{{{max_chars},}}", breaker, s)


def _risk_chart_image_path(prob: float) -> str:
    """
    Create a tiny bar chart image for the risk score and return a temp file path.
    """
    fig = plt.figure(figsize=(2.6, 0.8), dpi=200)
    plt.bar(["Risk"], [prob])
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _metrics_chart_image_path(inputs: Dict[str, float]) -> str:
    """
    Create a compact 'key metrics' chart image (glucose, BMI, age normalized),
    and return the temp file path.
    """
    # Normalize a few helpful metrics to a 0..1 scale for visualization
    glucose = float(inputs["Glucose"])
    bmi = float(inputs["BMI"])
    age = float(inputs["Age"])

    # Simple normalizations (heuristic ranges)
    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    glucose_n = norm(glucose, 70, 200)  # fasting-ish view
    bmi_n = norm(bmi, 18.5, 40)
    age_n = norm(age, 18, 80)

    labels = ["Glucose", "BMI", "Age"]
    values = [glucose_n, bmi_n, age_n]

    fig = plt.figure(figsize=(2.8, 1.6), dpi=200)
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1.0], ["Low", "Med", "High"])
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _make_pdf(
    patient_inputs: Dict[str, float],
    score: float,
    label: str,
    explanation: str,
    *,
    patient_name: str | None = None,
    case_id: str | None = None,
    logo_path: str | None = None,
) -> bytes:
    """
    Clean A4 portrait PDF in strict single-column layout:
      - Logo + header (Patient / Case ID)
      - Patient-entered metrics
      - Risk result + risk bar chart
      - Key metrics mini-chart
      - AI explanation (educational only)
      - Likely contributing factors
      - Next steps for care
    """
    pdf, default_font, bold_font, header_font, unicode_ok = _pdf_init_a4_unicode()

    # Margins & usable width
    left_margin, right_margin = 15, 15
    pdf.set_left_margin(left_margin)
    pdf.set_right_margin(right_margin)
    effective_width = pdf.w - left_margin - right_margin

    def block(
        text: str,
        *,
        lh: float = 6,
        font=None,
        top_gap: float = 0,
        pre_break: bool = True,
        align: str = "L",
    ):
        """Write a full-width paragraph block with safe wrapping and margin reset."""
        if top_gap:
            pdf.ln(top_gap)
        if font is None:
            font = default_font
        pdf.set_font(*font)
        txt = _sanitize_if_needed(text, unicode_ok)
        if pre_break:
            txt = _force_wrap_tokens(txt, max_chars=25)
        pdf.set_x(left_margin)
        pdf.multi_cell(effective_width, lh, txt, align=align)

    # Optional logo (top-right)
    if logo_path and Path(logo_path).exists():
        try:
            pdf.image(logo_path, x=pdf.w - right_margin - 28, y=15, w=28)
        except Exception:
            pass
    elif DEFAULT_LOGO.exists():
        try:
            pdf.image(str(DEFAULT_LOGO), x=pdf.w - right_margin - 28, y=15, w=28)
        except Exception:
            pass

    # Header
    pdf.set_font(*header_font)
    pdf.set_x(left_margin)
    pdf.multi_cell(effective_width, 10, "Diabetes Risk Assessment Report", align="L")
    pdf.set_font(default_font[0], default_font[1], 11)
    block(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", lh=7, pre_break=False)

    # Patient identity
    id_line = []
    if patient_name:
        id_line.append(f"Patient: {patient_name}")
    if case_id:
        id_line.append(f"Case ID: {case_id}")
    if id_line:
        block(" | ".join(id_line), lh=7, pre_break=False)
    pdf.ln(4)

    # Patient-entered Metrics
    pdf.set_font(bold_font[0], bold_font[1], 14)
    block("Patient-entered Metrics", lh=8, pre_break=False)
    pdf.set_font(*default_font)
    for k, v in patient_inputs.items():
        block(f"- {k}: {v}", lh=6)

    # Risk Result
    pdf.ln(4)
    pdf.set_font(bold_font[0], bold_font[1], 14)
    block("Risk Result", lh=8, pre_break=False)
    pdf.set_font(*default_font)
    block(f"Risk Category: {label}", lh=6, pre_break=False)
    block(f"Risk Score: {round(score * 100)}%", lh=6, pre_break=False)

    # Risk chart
    chart_path = _risk_chart_image_path(score)
    try:
        pdf.image(chart_path, x=left_margin, w=effective_width * 0.6)
    finally:
        try:
            os.unlink(chart_path)
        except Exception:
            pass

    # Key metrics chart
    metrics_chart_path = _metrics_chart_image_path(patient_inputs)
    try:
        pdf.image(metrics_chart_path, x=left_margin, w=effective_width * 0.75)
    finally:
        try:
            os.unlink(metrics_chart_path)
        except Exception:
            pass

    # AI Explanation
    pdf.ln(4)
    pdf.set_font(bold_font[0], bold_font[1], 14)
    block("AI Explanation (educational only)", lh=8, pre_break=False)
    pdf.set_font(*default_font)
    for paragraph in explanation.splitlines():
        block(paragraph, lh=6)  # token-wrapping inside block()

    # Likely Contributing Factors
    pdf.ln(2)
    pdf.set_font(bold_font[0], bold_font[1], 14)
    block("Likely Contributing Factors", lh=8, pre_break=False)
    pdf.set_font(*default_font)
    block(
        "Based on your provided metrics, the following factors may contribute to your current risk profile:\n"
        "- Fasting glucose level relative to typical ranges.\n"
        "- BMI near or above the healthy range.\n"
        "- Family history (as reflected by Diabetes Pedigree Function).\n"
        "These are general considerations and should be interpreted by a healthcare professional in context.",
        lh=6,
    )

    # Next Steps for Care
    pdf.ln(2)
    pdf.set_font(bold_font[0], bold_font[1], 14)
    block("Next Steps for Care", lh=8, pre_break=False)
    pdf.set_font(*default_font)
    block(
        "This report is for educational purposes only and is not a medical diagnosis. You may consider:\n"
        "1) Maintaining a balanced diet focused on whole foods and fiber.\n"
        "2) Engaging in at least 150 minutes of moderate activity per week.\n"
        "3) Monitoring fasting glucose and discussing HbA1c testing with your clinician.\n"
        "4) Scheduling periodic check-ups to review risk and prevention strategies.\n"
        "5) Seeking personalized guidance from a qualified healthcare professional.",
        lh=6,
    )

    return bytes(pdf.output(dest="S"))


# ---------- Tiny UI helpers ----------
def _chip(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0;
            border-radius:10px;
            background:#f6f6f6;
            border:1px solid #eaeaea;
            font-size:0.9rem;
        ">
            <b>{label}:</b> {value}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Page render ----------
def render() -> None:
    """Render the Diagnosis page."""
    st.header("Diabetes Diagnosis")
    st.write(
        "Enter your health metrics below to receive a risk assessment. "
        "All fields are required and should be numeric."
    )

    # 1) Load data (real CSV or synthetic fallback)
    df, source = load_diabetes_csv_or_synthesize()
    if source == "synthetic":
        st.warning("No diabetes.csv found. Using a small synthetic dataset for this session.")

    # 2) Train model
    model = train_model(df)

    # 3) Input form: header info + metrics
    with st.form("diag_form"):
        # Header fields
        colA, colB = st.columns([2, 1])
        patient_name = colA.text_input("Patient Name (optional)", value="")
        case_id = colB.text_input("Case ID (optional)", value="")
        logo_file = st.file_uploader("Upload Logo (PNG/JPG, optional)", type=["png", "jpg", "jpeg"])
        st.markdown("---")

        # Numeric inputs
        col1, col2, col3, col4 = st.columns(4)
        pregnancies = col1.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = col2.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
        blood_pressure = col3.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
        skin_thickness = col4.number_input("Skin Thickness", min_value=0, max_value=99, value=20, step=1)

        col5, col6, col7, col8 = st.columns(4)
        insulin = col5.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
        bmi = col6.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        dpf = col7.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
        age = col8.number_input("Age", min_value=0, max_value=120, value=30, step=1)

        submitted = st.form_submit_button("Diagnose")

    if not submitted:
        return

    # 4) Predict risk
    prob, label = predict_risk(
        model,
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
    )

    # On-screen visuals
    st.subheader("Diagnosis Result")
    st.progress(min(100, max(0, int(prob * 100))))
    _chip("Risk Score", f"{prob*100:.1f}%")
    _chip("Category", label)
    _chip("Glucose", str(glucose))
    _chip("BMI", f"{bmi:.1f}")
    _chip("Age", str(age))

    # 5) AI explanation (friendly & safe)
    prompt = (
        "Explain in simple, careful language why a person with these metrics "
        f"might be at '{label}' diabetes risk. Provide general educational guidance only. "
        f"Metrics: Pregnancies={pregnancies}, Glucose={glucose}, BP={blood_pressure}, "
        f"SkinThickness={skin_thickness}, Insulin={insulin}, BMI={bmi}, DPF={dpf}, Age={age}"
    )
    explanation = chat(
        prompt,
        system="You are a compassionate clinician. Avoid medical advice; keep it educational.",
        temperature=0.2,
    )

    st.markdown("### AI Explanation")
    st.write(explanation)

    # 6) Remember latest run for Results tab
    st.session_state["latest_prob"] = prob
    st.session_state["latest_label"] = label
    st.session_state["latest_inputs"] = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    st.session_state["latest_explanation"] = explanation

    # 7) Prepare logo file for PDF
    logo_path = None
    if logo_file is not None:
        tmp_logo = tempfile.NamedTemporaryFile(
            suffix=f".{logo_file.type.split('/')[-1]}",
            delete=False
        )
        tmp_logo.write(logo_file.getbuffer())
        tmp_logo.flush()
        logo_path = tmp_logo.name

    # 8) Build PDF with header info + logo + charts
    pdf_bytes = _make_pdf(
        st.session_state["latest_inputs"],
        prob,
        label,
        explanation,
        patient_name=patient_name.strip() or None,
        case_id=case_id.strip() or None,
        logo_path=logo_path,
    )

    # 9) Auto-save to ./reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (patient_name.strip().replace(" ", "_") or "report")
    pdf_path = REPORT_DIR / f"{safe_name}_{timestamp}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    st.success(f"Saved a copy to {pdf_path}")

    # 10) Download button
    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=f"diabetes_risk_{timestamp}.pdf",
        mime="application/pdf",
    )



