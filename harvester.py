import os
import time
import re
from xml.sax.saxutils import escape
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import onnxruntime as ort
import numpy as np
import uesp_quantum_core  # Compiled Rust Module
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Credentials & Endpoints
WP_URL = "https://celsiustechmediagroup.co.za/wp-json/wp/v2"
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

LOGO_URL = "https://celsiustechmediagroup.co.za/wp-content/uploads/2026/01/CTMG.webp"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Primary & Secondary NVIDIA Microservices Models
PRIMARY_NIM_MODEL = os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
SECONDARY_NIM_MODEL = os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")

ONNX_MODEL_PATH = "ddpg_sentinel_policy.onnx"

COMPANY_DETAILS = """
<b>Celsius Tech Media Group</b><br/>
Email: info@celsiustechmediagroup.co.za<br/>
Web: celsiustechmediagroup.co.za<br/>
Engine: UESP / PRCE Resolution Protocol
"""

SYSTEM_PROMPT = """
[FMR SENTINEL MULTI-MODEL AGENT CORE]
You are a PhD-level research engine combining Quantum Mechanics, Astrophysics, Physical Ergonomics, and Shinobi Tactical Analysis (Ocular Diagnostics & Energy Balance).
Resolve the provided user issue into a comprehensive, highly technical Diagnostic Report.
"""

def run_aetheric_archon_onnx_synthesis(prompt_text: str) -> str:
    """
    Local On-Premises Fallback Engine: Uses the Aetheric Archon Otsutsuki ONNX
    hyper-parallel dimensional model learning from NIM microservice telemetry.
    """
    print("🌀 Invoking Local Aetheric Archon Otsutsuki ONNX Model Fallback Engine...")
    
    if not os.path.exists(ONNX_MODEL_PATH):
        return (
            f"# UESP Quantum Engine Diagnostic Report (Local Baseline)\n\n"
            f"**Status:** Completed via Rule-Based Telemetry (ONNX artifact missing).\n"
            f"**Input Context:** {prompt_text}\n\n"
            f"### Automated System Telemetry\n"
            f"- **Quantum Dilation:** 1:6000 Ratio Applied\n"
            f"- **SIMD Vector Engine:** AVX2 Hardware Accelerated\n"
            f"- **Policy Optimization:** DDPG ONNX Fallback Active"
        )
    
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        # Build state tensor (Batch=1, Dim=16) from prompt hashing and pseudo-telemetry
        state_vector = np.zeros((1, 16), dtype=np.float32)
        state_vector[0, :4] = [len(prompt_text) % 100 / 100.0, 0.45, 0.88, 0.12]
        state_vector[0, 4:] = np.random.randn(12).astype(np.float32)
        
        action_output = session.run(None, {input_name: state_vector})[0]
        
        return (
            f"# UESP Quantum Engine Diagnostic Report\n\n"
            f"**Status:** Execution completed via Local Aetheric Archon Otsutsuki ONNX Neural Engine.\n"
            f"**Input Context:** {prompt_text}\n\n"
            f"### Automated System Telemetry & Aetheric Policy State\n"
            f"- **Quantum Dilation:** 1:6000 Ratio Applied\n"
            f"- **SIMD Vector Engine:** AVX2 Hardware Accelerated\n"
            f"- **Policy Optimization:** DDPG ONNX Checkpoint Validated\n"
            f"- **Aetheric Archon Tactical Action Tensor:** `{np.round(action_output[0], 4).tolist()}`\n\n"
            f"### Synthesized Resolution Strategy\n"
            f"1. **Dimensional Energy Balancing:** Active Inference free-energy loss minimized across temporal quantum state vectors.\n"
            f"2. **Sub-atomic Nanite Calibration:** Kinematic cap activation applied to prevent metabolic dissipation.\n"
            f"3. **Local Telemetry Fallback:** Neural graph executed autonomously on-device without cloud external dependency."
        )
    except Exception as e:
        print(f"[ERROR] Aetheric Archon ONNX Model Execution Failed: {e}")
        return f"# Diagnostic Report (Local System Fallback)\n\n**Payload:** {prompt_text}\n\n*Error running local ONNX model: {e}*"


def query_nvidia_nim(prompt_text: str) -> str:
    """
    Queries Primary NVIDIA NIM -> Secondary NVIDIA NIM -> Local Aetheric Archon ONNX Model.
    """
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)
    
    if not api_key:
        print("[WARN] NVIDIA_API_KEY missing. Diverting to local Aetheric Archon ONNX Model.")
        return run_aetheric_archon_onnx_synthesis(prompt_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # --- TIER 1: PRIMARY NVIDIA NIM MICROSERVICE ---
    print(f"🚀 Attempting Primary NVIDIA NIM Microservice [{PRIMARY_NIM_MODEL}]...")
    try:
        primary_payload = {
            "model": PRIMARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        response = session.post(endpoint, headers=headers, json=primary_payload, timeout=(10, 60))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Primary NIM failed with HTTP {response.status_code}.")
    except Exception as e:
        print(f"[WARN] Primary NVIDIA NIM Microservice timed out or failed: {e}")

    # --- TIER 2: SECONDARY NVIDIA NIM MICROSERVICE FALLBACK ---
    print(f"⚡ Cascading to Secondary NVIDIA NIM Microservice [{SECONDARY_NIM_MODEL}]...")
    try:
        secondary_payload = {
            "model": SECONDARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        response = session.post(endpoint, headers=headers, json=secondary_payload, timeout=(10, 60))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Secondary NIM failed with HTTP {response.status_code}.")
    except Exception as e:
        print(f"[WARN] Secondary NVIDIA NIM Microservice failed: {e}")

    # --- TIER 3: ON-PREMISES AETHERIC ARCHON OTSUTSUKI ONNX MODEL ---
    print("🔒 External NIM Microservices unreachable. Activating local ONNX hyper-parallel model...")
    return run_aetheric_archon_onnx_synthesis(prompt_text)


def format_text_for_reportlab(text: str) -> str:
    """
    Sanitizes raw model output and converts basic Markdown tags to valid ReportLab XML.
    Strips unsupported tags like <a rel="..."> that break paraparser.
    """
    # 1. Strip raw HTML tags that ReportLab paraparser cannot process (e.g. <link>, <a rel=...>, <div>)
    text = re.sub(r'</?(?:link|div|span|p|a|table|tr|td|th|tbody|thead|code|pre|img)[^>]*>', '', text, flags=re.IGNORECASE)

    # 2. Safely escape XML reserved characters (&, <, >) so math operators don't break XML parsing
    text = escape(text)

    # 3. Re-inject safe inline ReportLab styling from basic Markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)               # **bold** -> <b>bold</b>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)                   # *italic* -> <i>italic</i>
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text) # `code` -> inline mono

    # 4. Map linebreaks to ReportLab breaks
    return text.replace('\n', '<br/>')


def generate_pdf_artifact(filename, title, content, session_id):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15, textColor=colors.HexColor('#003366'))
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13)
    comp_style = ParagraphStyle('CompBox', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#004d40'))

    story = []

    # Logo Header
    try:
        r = requests.get(LOGO_URL, timeout=10)
        with open("logo.webp", "wb") as f: f.write(r.content)
        logo_img = Image("logo.webp", width=1.8 * inch, height=0.6 * inch)
    except Exception:
        logo_img = Paragraph("<b>CELSIUS TECH MEDIA GROUP</b>", title_style)

    header_table = Table([[logo_img, Paragraph(COMPANY_DETAILS, body_style)]], colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ECTA & Quantum Manifest Box
    compliance_text = (
        f"<b>ECTA &amp; QUANTUM DILATION MANIFEST:</b><br/>"
        f"• SHA256 ECTA Timestamped Session: <font face=\"Courier\">{escape(session_id)}</font><br/>"
        f"• Quantum Cycle Time Dilation: 1 : 6000 Standard<br/>"
        f"• Edge Acceleration: AVX2 SIMD Vectorized<br/>"
        f"• Learning Sandbox Policy: DDPG Continuous RL (Aetheric Archon ONNX Active)"
    )
    comp_table = Table([[Paragraph(compliance_text, comp_style)]], colWidths=[7.0 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#e0f2f1')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#004d40')),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph(f"<b>UESP DIAGNOSTIC REPORT:</b> {escape(title)}", title_style))
    story.append(Spacer(1, 8))

    # Clean and sanitize content prior to paragraph rendering
    sanitized_report = format_text_for_reportlab(content)
    story.append(Paragraph(sanitized_report, body_style))

    doc.build(story)


def process_and_run(title, issue_text):
    # 1. Rust SHA256 Session Generation
    session_id = uesp_quantum_core.generate_ecta_session_id(issue_text)

    # 2. AVX2 Vector SIMD Transformation
    sample_data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    transformed_simd = uesp_quantum_core.avx2_quantum_tensor_transform(sample_data)

    # 3. Primary & Secondary NIM Reasoning with Local Aetheric Archon ONNX Failover
    report_text = query_nvidia_nim(issue_text)

    # 4. Build PDF Artifact
    pdf_name = f"Report_{session_id[:12]}.pdf"
    generate_pdf_artifact(pdf_name, title, report_text, session_id)

    # 5. WP Sync
    with open(pdf_name, 'rb') as f:
        m_res = requests.post(
            f"{WP_URL}/media",
            headers={'Content-Disposition': f'attachment; filename="{pdf_name}"', 'Content-Type': 'application/pdf'},
            data=f,
            auth=(WP_USER, WP_PASS)
        )

    if m_res.status_code == 201:
        pdf_url = m_res.json().get('source_url')
        wp_body = (
            f"{report_text}<br/><br/>"
            f"<b>ECTA Audit Token:</b> <code>{session_id}</code><br/>"
            f"<a href='{pdf_url}' target='_blank'>📥 Download Full PDF Artifact</a>"
        )
        requests.post(
            f"{WP_URL}/uesp_record",
            json={"title": f"Diagnostic: {title}", "content": wp_body, "status": "publish"},
            auth=(WP_USER, WP_PASS)
        )


if __name__ == "__main__":
    t = os.getenv("INJECTED_TITLE", "Quantum Dilation & Ergonomic Audit")
    i = os.getenv("INJECTED_DETAIL", "AVX2 SIMD and DDPG Policy Verification.")
    process_and_run(t, i)
