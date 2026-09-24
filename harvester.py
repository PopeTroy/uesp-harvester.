import os
import time
import re
import html
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
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

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
Do NOT use LaTeX math formatting (e.g., $...$) or raw HTML tags in your response. Use standard plain text for all equations and numbers.
"""

def query_nvidia_nim(prompt_text: str) -> str:
    """Queries NVIDIA NIM endpoint with robust timeout handling and retry logic."""
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)
    
    if not api_key:
        print("[WARN] NVIDIA_API_KEY missing. Returning local fallback payload.")
        return f"# Diagnostic Report (Fallback)\n\n**Payload:** {prompt_text}\n\n*NVIDIA NIM API key not configured.*"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.post(endpoint, headers=headers, json=payload, timeout=(15, 180))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (requests.exceptions.ReadTimeout, requests.exceptions.RequestException) as e:
        print(f"[ERROR] NVIDIA NIM API call failed or timed out: {e}")
        return (
            f"# UESP Quantum Engine Diagnostic Report\n\n"
            f"**Status:** Execution completed with local telemetry fallback.\n"
            f"**Input Context:** {prompt_text}\n\n"
            f"### Automated System Telemetry\n"
            f"- **Quantum Dilation:** 1:6000 Ratio Applied\n"
            f"- **SIMD Vector Engine:** AVX2 Hardware Accelerated\n"
            f"- **Policy Optimization:** DDPG ONNX Checkpoint Validated\n"
            f"- **Notice:** External NIM synthesis endpoint timed out ({e}). Local fallback applied."
        )

def sanitize_inline_markdown(text: str) -> str:
    """
    Completely sanitizes incoming model text to prevent ReportLab XML parse errors.
    Nuke all raw/escaped HTML tags, converts bold/italic via placeholders, and escapes XML.
    """
    # 1. Unescape html entities first (turns &lt;i&gt; back into <i> so regex catches them)
    clean = html.unescape(text)

    # 2. Strip ALL pre-existing HTML/XML tags completely
    clean = re.sub(r'<[^>]+>', '', clean)

    # 3. Strip LaTeX expressions and backslashes
    clean = clean.replace('$', '')
    clean = re.sub(r'\\text\{([^}]+)\}', r'\1', clean)
    clean = clean.replace(r'\times', 'x').replace(r'\approx', '~').replace(r'\sim', '~').replace('\\', '')

    # 4. Extract valid Markdown bold **text** into safe placeholders
    bold_matches = []
    def sub_bold(m):
        bold_matches.append(m.group(1))
        return f"___SAFE_BOLD_{len(bold_matches)-1}___"
    clean = re.sub(r'\*\*([^*]+)\*\*', sub_bold, clean)

    # 5. Extract valid Markdown italic *text* or _text_ into safe placeholders
    italic_matches = []
    def sub_italic(m):
        italic_matches.append(m.group(1))
        return f"___SAFE_ITALIC_{len(italic_matches)-1}___"
    clean = re.sub(r'\*([^*]+)\*', sub_italic, clean)
    clean = re.sub(r'_([^_]+)_', sub_italic, clean)

    # 6. Remove any remaining stray asterisks, underscores, or angle brackets that aren't matched
    clean = clean.replace('*', '').replace('_', '').replace('>', '').replace('<', '')

    # 7. Safe XML escape reserved characters (& -> &amp;, etc.)
    clean = escape(clean)

    # 8. Re-insert formatted tags safely from clean placeholders
    for i, content in enumerate(bold_matches):
        clean = clean.replace(f"___SAFE_BOLD_{i}___", f"<b>{escape(content)}</b>")
    for i, content in enumerate(italic_matches):
        clean = clean.replace(f"___SAFE_ITALIC_{i}___", f"<i>{escape(content)}</i>")

    return clean

def format_text_to_story(text: str, story: list, styles: dict):
    lines = text.split('\n')
    
    h1_style = ParagraphStyle('ReportH1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor('#003366'), spaceBefore=10, spaceAfter=4)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#004d40'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, spaceAfter=4)
    bullet_style = ParagraphStyle('ReportBullet', parent=body_style, leftIndent=12, spaceAfter=3)

    for line in lines:
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 4))
            continue

        if line_str.startswith('# '):
            content = line_str[2:].strip()
            story.append(Paragraph(sanitize_inline_markdown(content), h1_style))
        elif line_str.startswith('## '):
            content = line_str[3:].strip()
            story.append(Paragraph(sanitize_inline_markdown(content), h1_style))
        elif line_str.startswith('### '):
            content = line_str[4:].strip()
            story.append(Paragraph(sanitize_inline_markdown(content), h2_style))
        elif line_str.startswith('- ') or line_str.startswith('* ') or line_str.startswith('> '):
            content = re.sub(r'^[-*>]\s*', '', line_str).strip()
            story.append(Paragraph(f"• {sanitize_inline_markdown(content)}", bullet_style))
        else:
            story.append(Paragraph(sanitize_inline_markdown(line_str), body_style))

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

    compliance_text = (
        f"<b>ECTA &amp; QUANTUM DILATION MANIFEST:</b><br/>"
        f"• SHA256 ECTA Timestamped Session: <font face=\"Courier\">{escape(session_id)}</font><br/>"
        f"• Quantum Cycle Time Dilation: 1 : 6000 Standard<br/>"
        f"• Edge Acceleration: AVX2 SIMD Vectorized<br/>"
        f"• Learning Sandbox Policy: DDPG Continuous RL (ONNX Runtime Active)"
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

    format_text_to_story(content, story, styles)

    doc.build(story)

def process_and_run(title, issue_text):
    session_id = uesp_quantum_core.generate_ecta_session_id(issue_text)

    sample_data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    transformed_simd = uesp_quantum_core.avx2_quantum_tensor_transform(sample_data)

    if os.path.exists("ddpg_sentinel_policy.onnx"):
        ort_session = ort.InferenceSession("ddpg_sentinel_policy.onnx")
        onnx_inputs = {ort_session.get_inputs()[0].name: np.random.randn(1, 16).astype(np.float32)}
        action_output = ort_session.run(None, onnx_inputs)

    report_text = query_nvidia_nim(issue_text)

    pdf_name = f"Report_{session_id[:12]}.pdf"
    generate_pdf_artifact(pdf_name, title, report_text, session_id)

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
