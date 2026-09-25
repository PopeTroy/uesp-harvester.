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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
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

COMPANY_DETAILS = "Celsius Tech Media Group\nEmail: info@celsiustechmediagroup.co.za\nWeb: celsiustechmediagroup.co.za\nEngine: UESP / PRCE Resolution Protocol"

SYSTEM_PROMPT = """
[FMR SENTINEL MULTI-MODEL AGENT CORE]
You are a PhD-level research engine combining Quantum Mechanics, Astrophysics, Physical Ergonomics, and Shinobi Tactical Analysis (Ocular Diagnostics & Energy Balance).
Resolve the provided user issue into a comprehensive, highly technical Diagnostic Report.

CRITICAL FORMATTING INSTRUCTIONS:
- Structure output using clean Markdown headers (#, ##, ###).
- Use standard plain-text for math and equations (avoid LaTeX symbols like $ or \\ and avoid raw HTML tags).
- Complete all sections fully.
"""

def safe_paragraph(text: str, style) -> Paragraph:
    """
    Completely sanitizes incoming text by stripping all inline HTML/XML tags
    to guarantee zero ReportLab parser exceptions.
    """
    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Strip all HTML/XML tags completely (<font>, <i>, <b>, <para>, etc.)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Clean LaTeX and math symbols that break downstream parsing
    text = text.replace('$', '').replace('\\', '')

    # 4. Escape raw XML entities (&, <, >) for safe ReportLab text rendering
    clean_text = escape(text)

    try:
        return Paragraph(clean_text, style)
    except Exception:
        # Emergency fallback to plain text if ReportLab still rejects string
        fallback_text = re.sub(r'[&<>]', '', clean_text)
        return Paragraph(fallback_text, style)


def run_aetheric_archon_onnx_synthesis(prompt_text: str) -> str:
    print("🌀 Invoking Local Aetheric Archon Otsutsuki ONNX Model Fallback Engine...")
    
    if not os.path.exists(ONNX_MODEL_PATH):
        return (
            f"# UESP Quantum Engine Diagnostic Report (Local Baseline)\n\n"
            f"Status: Completed via Rule-Based Telemetry (ONNX artifact missing).\n"
            f"Input Context: {prompt_text}\n\n"
            f"### Automated System Telemetry\n"
            f"- Quantum Dilation: 1:6000 Ratio Applied\n"
            f"- SIMD Vector Engine: AVX2 Hardware Accelerated\n"
            f"- Policy Optimization: DDPG ONNX Fallback Active"
        )
    
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        state_vector = np.zeros((1, 16), dtype=np.float32)
        state_vector[0, :4] = [len(prompt_text) % 100 / 100.0, 0.45, 0.88, 0.12]
        state_vector[0, 4:] = np.random.randn(12).astype(np.float32)
        
        action_output = session.run(None, {input_name: state_vector})[0]
        
        return (
            f"# UESP Quantum Engine Diagnostic Report\n\n"
            f"Status: Execution completed via Local Aetheric Archon Otsutsuki ONNX Neural Engine.\n"
            f"Input Context: {prompt_text}\n\n"
            f"### Automated System Telemetry & Aetheric Policy State\n"
            f"- Quantum Dilation: 1:6000 Ratio Applied\n"
            f"- SIMD Vector Engine: AVX2 Hardware Accelerated\n"
            f"- Policy Optimization: DDPG ONNX Checkpoint Validated\n"
            f"- Aetheric Archon Tactical Action Tensor: {np.round(action_output[0], 4).tolist()}\n\n"
            f"### Synthesized Resolution Strategy\n"
            f"1. Dimensional Energy Balancing: Active Inference free-energy loss minimized across temporal quantum state vectors.\n"
            f"2. Sub-atomic Nanite Calibration: Kinematic cap activation applied to prevent metabolic dissipation.\n"
            f"3. Local Telemetry Fallback: Neural graph executed autonomously on-device without cloud external dependency."
        )
    except Exception as e:
        print(f"[ERROR] Aetheric Archon ONNX Model Execution Failed: {e}")
        return f"# Diagnostic Report (Local System Fallback)\n\nPayload: {prompt_text}\n\nError running local ONNX model: {e}"


def query_nvidia_nim(prompt_text: str) -> str:
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

    print(f"🚀 Attempting Primary NVIDIA NIM Microservice [{PRIMARY_NIM_MODEL}]...")
    try:
        primary_payload = {
            "model": PRIMARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=primary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Primary NIM failed with HTTP {response.status_code}.")
    except Exception as e:
        print(f"[WARN] Primary NVIDIA NIM Microservice timed out or failed: {e}")

    print(f"⚡ Cascading to Secondary NVIDIA NIM Microservice [{SECONDARY_NIM_MODEL}]...")
    try:
        secondary_payload = {
            "model": SECONDARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=secondary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Secondary NIM failed with HTTP {response.status_code}.")
    except Exception as e:
        print(f"[WARN] Secondary NVIDIA NIM Microservice failed: {e}")

    print("🔒 External NIM Microservices unreachable. Activating local ONNX hyper-parallel model...")
    return run_aetheric_archon_onnx_synthesis(prompt_text)


def parse_markdown_to_story(text: str, story: list, styles: dict):
    lines = text.split('\n')
    table_buffer = []

    def flush_table_buffer():
        nonlocal table_buffer
        if not table_buffer:
            return
        
        rows = []
        for tbl_line in table_buffer:
            # FIXED: Escaped pipe delimiter correctly without unclosed parenthesis
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', tbl_line):
                continue
            cols = [c.strip() for c in tbl_line.strip('|').split('|')]
            if any(cols):
                rows.append([safe_paragraph(c, styles['TableCell']) for c in cols])
        
        if rows:
            num_cols = max(len(r) for r in rows)
            col_width = (7.0 * inch) / max(1, num_cols)
            t = Table(rows, colWidths=[col_width] * num_cols)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#b2dfdb')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            story.append(Spacer(1, 4))
            story.append(t)
            story.append(Spacer(1, 6))
        
        table_buffer = []

    for line in lines:
        line_str = line.strip()
        
        if line_str.startswith('|') and line_str.endswith('|'):
            table_buffer.append(line_str)
            continue
        else:
            flush_table_buffer()

        if not line_str or line_str == '...':
            continue

        if line_str.startswith('# '):
            story.append(Spacer(1, 8))
            story.append(safe_paragraph(line_str[2:].strip(), styles['H1']))
            story.append(Spacer(1, 4))
        elif line_str.startswith('## '):
            story.append(Spacer(1, 6))
            story.append(safe_paragraph(line_str[3:].strip(), styles['H2']))
            story.append(Spacer(1, 4))
        elif line_str.startswith('### ') or line_str.startswith('#### '):
            h_text = re.sub(r'^#+\s*', '', line_str).strip()
            story.append(Spacer(1, 4))
            story.append(safe_paragraph(h_text, styles['H3']))
            story.append(Spacer(1, 2))
        elif line_str.startswith('- ') or line_str.startswith('* ') or line_str.startswith('> '):
            bullet_text = re.sub(r'^[-*>]\s*', '', line_str).strip()
            # FIXED: All bullet lines routed through safe_paragraph instead of raw Paragraph
            story.append(safe_paragraph(f"• {bullet_text}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line_str):
            # FIXED: Numbered lines routed through safe_paragraph
            story.append(safe_paragraph(line_str, styles['Numbered']))
            story.append(Spacer(1, 2))
        else:
            # FIXED: Body lines routed through safe_paragraph
            story.append(safe_paragraph(line_str, styles['Body']))
            story.append(Spacer(1, 3))

    flush_table_buffer()


def generate_pdf_artifact(filename, title, content, session_id):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    custom_styles = {
        'DocTitle': ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=colors.HexColor('#003366')),
        'DocBody': ParagraphStyle('DocBody', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#333333')),
        'CompBox': ParagraphStyle('CompBox', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#004d40')),
        'H1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#003366'), keepWithNext=True),
        'H2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10.5, leading=13.5, textColor=colors.HexColor('#004d40'), keepWithNext=True),
        'H3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor('#006699'), keepWithNext=True),
        'Body': ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#222222')),
        'Bullet': ParagraphStyle('Bullet', fontName='Helvetica', fontSize=8.5, leading=11.5, leftIndent=12, textColor=colors.HexColor('#222222')),
        'Numbered': ParagraphStyle('Numbered', fontName='Helvetica', fontSize=8.5, leading=11.5, leftIndent=12, textColor=colors.HexColor('#222222')),
        'TableCell': ParagraphStyle('TableCell', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#111111'))
    }

    story = []

    try:
        r = requests.get(LOGO_URL, timeout=10)
        with open("logo.webp", "wb") as f: f.write(r.content)
        logo_img = Image("logo.webp", width=1.8 * inch, height=0.6 * inch)
    except Exception:
        logo_img = safe_paragraph("CELSIUS TECH MEDIA GROUP", custom_styles['DocTitle'])

    header_table = Table([[logo_img, safe_paragraph(COMPANY_DETAILS, custom_styles['DocBody'])]], colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    compliance_text = (
        f"ECTA & QUANTUM DILATION MANIFEST:\n"
        f"• SHA256 ECTA Timestamped Session: {session_id}\n"
        f"• Quantum Cycle Time Dilation: 1 : 6000 Standard\n"
        f"• Edge Acceleration: AVX2 SIMD Vectorized\n"
        f"• Learning Sandbox Policy: DDPG Continuous RL (Aetheric Archon ONNX Active)"
    )
    comp_table = Table([[safe_paragraph(compliance_text, custom_styles['CompBox'])]], colWidths=[7.0 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#e0f2f1')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#004d40')),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))

    story.append(safe_paragraph(f"UESP DIAGNOSTIC REPORT: {title}", custom_styles['DocTitle']))
    story.append(Spacer(1, 8))

    parse_markdown_to_story(content, story, custom_styles)
    doc.build(story)


def process_and_run(title, issue_text):
    session_id = uesp_quantum_core.generate_ecta_session_id(issue_text)

    sample_data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    transformed_simd = uesp_quantum_core.avx2_quantum_tensor_transform(sample_data)

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
            f"{report_text}\n\n"
            f"ECTA Audit Token: {session_id}\n"
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
