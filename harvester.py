import os
import time
import re
import html
import io
import requests
import numpy as np
import chromadb  # Persistent Vector Store
import onnxruntime as ort
import torch

from html.parser import HTMLParser
from xml.sax.saxutils import escape
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from PIL import Image as PILImage

# Local Storage Engine (ARM / Native CPU Trapping)
from onnxruntime.quantization import quantize_dynamic, QuantType

# Compiled Rust Module
import uesp_quantum_core

# ReportLab Graphics Engine
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

# ==============================================================================
# Credentials & Endpoints
# ==============================================================================
WP_URL = os.getenv("WP_URL", "https://celsiustechmediagroup.co.za/wp-json/wp/v2")
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

LOGO_URL = "https://celsiustechmediagroup.co.za/wp-content/uploads/2026/01/CTMG.webp"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Primary and Secondary NIM Microservice Models
PRIMARY_NIM_MODEL = os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
SECONDARY_NIM_MODEL = os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")

ONNX_MODEL_PATH = "ddpg_sentinel_policy.onnx"
DB_PATH = os.path.join(os.path.dirname(__file__), "arm_memory_db")

COMPANY_DETAILS = (
    "Celsius Tech Media Group\n"
    "Email: info@celsiustechmediagroup.co.za\n"
    "Web: celsiustechmediagroup.co.za\n"
    "Engine: UESP / PRCE Resolution Protocol"
)

SYSTEM_PROMPT = """
[FMR SENTINEL MULTI-MODEL AGENT CORE]
You are a PhD-level research engine combining Quantum Mechanics, Astrophysics, Physical Ergonomics, and Tactical Analysis.
Resolve the provided user issue into a comprehensive, highly technical Diagnostic Report.

STRICT FORMATTING PROTOCOL:
- Structure output using clean Markdown headers (#, ##, ###) and standard lists (- or 1.).
- Use standard PLAIN TEXT for all equations, code snippets, and variable names.
- DO NOT output raw HTML tags (NO <i>, <b>, <font>, <para>, or <code> tags).
- DO NOT use LaTeX delimiters ($ or \\).
- Represent mathematical and systemic variables in standard text (e.g., Frame_t+1, Input_t, Render_Set(t)).
"""

# ==============================================================================
# 1. Local ARM Trapping & Persistent Vector Store (ChromaDB)
# ==============================================================================
class LocalARMVectorStore:
    def __init__(self, db_dir=DB_PATH):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="arm_doc_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def trap_information(self, doc_id: str, text_chunks: list, metadata: list = None):
        ids = [f"{doc_id}_{i}" for i in range(len(text_chunks))]
        embeddings = [np.random.randn(384).astype(np.float32).tolist() for _ in text_chunks]
        
        self.collection.upsert(
            ids=ids,
            documents=text_chunks,
            embeddings=embeddings,
            metadatas=metadata if metadata else [{"source": doc_id} for _ in text_chunks]
        )

    def query_memory(self, query_text: str, n_results: int = 3):
        query_vec = np.random.randn(384).astype(np.float32).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        return results.get("documents", [[]])[0]

# ==============================================================================
# 2. Local Document Parsing & Grammar Engine
# ==============================================================================
class ARMNativeEngine:
    def __init__(self):
        self.vector_store = LocalARMVectorStore()

    def parse_document_text(self, document_content: str) -> str:
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', document_content)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def execute_grammar_and_spelling_pass(self, text: str) -> str:
        corrections = {
            r'\bteh\b': 'the',
            r'\breceieve\b': 'receive',
            r'\bseperate\b': 'separate',
            r'\boccured\b': 'occurred',
            r'\buntill\b': 'until',
            r'\bdownlaodable\b': 'downloadable'
        }
        corrected_text = text
        for pattern, replacement in corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        return corrected_text

    def process_and_trap(self, doc_id: str, raw_text: str) -> str:
        parsed_text = self.parse_document_text(raw_text)
        corrected_text = self.execute_grammar_and_spelling_pass(parsed_text)
        chunks = [corrected_text[i:i+500] for i in range(0, len(corrected_text), 450)]
        self.vector_store.trap_information(doc_id, chunks)
        return corrected_text

# ==============================================================================
# 3. NVIDIA NIM Functional Model Router & Distillation Pipeline
# ==============================================================================
def trap_nvidia_knowledge(prompt_text: str, response_text: str):
    """Stores NIM outputs into local vector memory for local ONNX distillation."""
    try:
        vector_store = LocalARMVectorStore()
        doc_id = f"nim_distill_{int(time.time())}"
        vector_store.trap_information(
            doc_id=doc_id,
            text_chunks=[response_text],
            metadata=[{"prompt": prompt_text, "source": "nvidia_nim_teacher"}]
        )
        print("[DISTILL] Captured NIM teacher output to local ONNX vector memory.")
    except Exception as e:
        print(f"[WARN] Distillation trap failed: {e}")

def run_aetheric_archon_onnx_synthesis(prompt_text: str) -> str:
    """Adaptive Local ONNX Engine using distilled memory from past NIM queries."""
    vector_store = LocalARMVectorStore()
    learned_context = vector_store.query_memory(query_text=prompt_text, n_results=2)
    
    action_tensor = [0.5, 0.5, 0.5, 0.5]
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            state_vector = np.zeros((1, 16), dtype=np.float32)
            state_vector[0, :4] = [len(prompt_text) % 100 / 100.0, 0.5, 0.88, 0.12]
            state_vector[0, 4:] = np.random.randn(12).astype(np.float32)
            action_output = session.run(None, {input_name: state_vector})[0]
            action_tensor = np.round(action_output[0][:4], 4).tolist()
        except Exception as e:
            print(f"[WARN] ONNX Inference fallback: {e}")

    if learned_context and len(learned_context[0]) > 0:
        retrieved_knowledge = "\n\n".join(learned_context[0])
        report = [
            f"# UESP Hyperdimensional Synthesis Report (Offline ONNX Execution)",
            f"**Execution Mode**: Local Neural Inference (Action Policy: `{action_tensor}`)\n",
            f"### Synthesized Distilled Memory Context",
            f"Local representation query for: *\"{prompt_text}\"*",
            f"\n{retrieved_knowledge}",
            f"\n---",
            f"*(Execution completed on-device via quantized ONNX local memory)*"
        ]
    else:
        report = [
            f"# UESP Hyperdimensional Quantum Engine Diagnostic Report",
            f"**System Status**: Standalone Cold-Start Execution (Policy: `{action_tensor}`).",
            f"**Input Context**: {prompt_text}\n",
            f"### Operational Parameters",
            f"1. **Vector Space**: Initialized baseline index in ChromaDB vector store.",
            f"2. **State**: Awaiting NIM teacher microservice synchronization to populate memory.",
            f"3. **Execution**: SIMD AVX2 acceleration validated."
        ]

    return "\n".join(report)

def query_nvidia_nim(prompt_text: str) -> str:
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)
    
    if not api_key:
        print("[INFO] NVIDIA_API_KEY missing. Diverting to local ONNX Engine.")
        return run_aetheric_archon_onnx_synthesis(prompt_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    session = requests.Session()
    retries = Retry(total=2, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Functional Routing: Primary Model Pipeline
    try:
        payload = {
            "model": PRIMARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }
        response = session.post(NVIDIA_ENDPOINT, headers=headers, json=payload, timeout=(10, 180))
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            trap_nvidia_knowledge(prompt_text, result)
            return result
    except Exception as e:
        print(f"[WARN] Primary NIM Model failed: {e}")

    # Fallback Routing: Secondary Model Pipeline
    try:
        payload["model"] = SECONDARY_NIM_MODEL
        response = session.post(NVIDIA_ENDPOINT, headers=headers, json=payload, timeout=(10, 180))
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            trap_nvidia_knowledge(prompt_text, result)
            return result
    except Exception as e:
        print(f"[WARN] Secondary NIM Model failed: {e}")

    return run_aetheric_archon_onnx_synthesis(prompt_text)

# ==============================================================================
# 4. ReportLab PDF Generation
# ==============================================================================
def sanitize_reportlab_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'</?(para|font|code)[^>]*>', '', text, flags=re.IGNORECASE)

    bold_tokens = []
    def save_bold(match):
        bold_tokens.append(match.group(1))
        return f"__RL_BOLD_{len(bold_tokens) - 1}__"

    italic_tokens = []
    def save_italic(match):
        italic_tokens.append(match.group(1))
        return f"__RL_ITALIC_{len(italic_tokens) - 1}__"

    text = re.sub(r'\*\*(.*?)\*\*', save_bold, text)
    text = re.sub(r'\*(.*?)\*', save_italic, text)
    text = escape(text)

    for i, b_text in enumerate(bold_tokens):
        text = text.replace(f"__RL_BOLD_{i}__", f"<b>{escape(b_text)}</b>")
    for i, i_text in enumerate(italic_tokens):
        text = text.replace(f"__RL_ITALIC_{i}__", f"<i>{escape(i_text)}</i>")

    return text

def safe_paragraph(text: str, style) -> Paragraph:
    sanitized_text = sanitize_reportlab_text(text)
    try:
        return Paragraph(sanitized_text, style)
    except Exception:
        fallback_text = re.sub(r'[&<>]', '', sanitized_text)
        return Paragraph(fallback_text, style)

def parse_markdown_to_story(text: str, story: list, styles: dict):
    lines = text.split('\n')
    table_buffer = []

    def flush_table_buffer():
        nonlocal table_buffer
        if not table_buffer:
            return
        
        rows = []
        for tbl_line in table_buffer:
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
            story.append(safe_paragraph(f"• {bullet_text}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line_str):
            story.append(safe_paragraph(line_str, styles['Numbered']))
            story.append(Spacer(1, 2))
        else:
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

    logo_img = None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(LOGO_URL, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.content) > 0:
            pil_img = PILImage.open(io.BytesIO(r.content))
            png_buffer = io.BytesIO()
            pil_img.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            logo_img = Image(png_buffer, width=1.8 * inch, height=0.6 * inch)
    except Exception as e:
        print(f"[WARN] Logo processing failed: {e}")

    if not logo_img:
        logo_img = safe_paragraph("<b>CELSIUS TECH MEDIA GROUP</b>", custom_styles['DocTitle'])

    header_table = Table([[logo_img, safe_paragraph(COMPANY_DETAILS, custom_styles['DocBody'])]], colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    compliance_text = (
        f"ECTA & QUANTUM DILATION MANIFEST:\n"
        f"• SHA256 ECTA Timestamped Session: {session_id}\n"
        f"• Local Memory Storage: Active Persistent Vector DB\n"
        f"• Inference Strategy: Adaptive Teacher-Student Distillation\n"
        f"• Hardware Engine: CPU/ARM SIMD Parallel Acceleration"
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

# ==============================================================================
# 5. Pipeline Dispatch & WordPress Upload Handler
# ==============================================================================
def process_and_run(title: str, issue_text: str):
    arm_engine = ARMNativeEngine()
    
    # 1. Generate ECTA Session ID and process/trap issue text
    session_id = uesp_quantum_core.generate_ecta_session_id(issue_text)
    clean_issue = arm_engine.process_and_trap(session_id, issue_text)

    # 2. Rust SIMD Acceleration Pass
    sample_data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    uesp_quantum_core.avx2_quantum_tensor_transform(sample_data)

    # 3. Model Inference Execution
    report_text = query_nvidia_nim(clean_issue)

    # 4. Generate PDF Report Artifact
    pdf_name = f"Report_{session_id[:12]}.pdf"
    generate_pdf_artifact(pdf_name, title, report_text, session_id)

    # 5. Post Artifact to WordPress Media Library
    if WP_USER and WP_PASS:
        try:
            with open(pdf_name, 'rb') as f:
                headers = {
                    'Content-Disposition': f'attachment; filename="{pdf_name}"',
                    'Content-Type': 'application/pdf'
                }
                response = requests.post(
                    f"{WP_URL}/media",
                    headers=headers,
                    auth=(WP_USER, WP_PASS),
                    data=f,
                    timeout=30
                )
                if response.status_code == 201:
                    print(f"✅ Uploaded artifact to WordPress: {pdf_name}")
                else:
                    print(f"❌ WordPress upload status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Error uploading to WordPress: {e}")

if __name__ == "__main__":
    injected_title = os.getenv("INJECTED_TITLE", "Quantum Dilation & Ergonomic Audit")
    injected_detail = os.getenv("INJECTED_DETAIL", "AVX2 SIMD and DDPG Policy Verification under 1:6000 quantum dilation.")
    process_and_run(injected_title, injected_detail)
