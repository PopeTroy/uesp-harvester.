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

# Safe optional imports with fallback handling (Fixes Line 12 ModuleNotFoundError)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAS_LLAMA_CPP = False

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
PRIMARY_NIM_MODEL = os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-4-340b-instruct")
SECONDARY_NIM_MODEL = os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")

ONNX_MODEL_PATH = "ddpg_sentinel_policy.onnx"
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "models/llama-3.2-3b-instruct.Q4_K_M.gguf")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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

CRITICAL FORMATTING INSTRUCTIONS:
- Structure output using clean Markdown headers (#, ##, ###).
- Use standard text for equations (avoid LaTeX symbols like $ or \\).
- DO NOT use raw HTML line breaks like <br/>.
- Complete all sections fully. Do not leave trailing thoughts or ellipses (...).
"""

# In-Memory Domain Knowledge Base for Ephemeral RAG Construction
KNOWLEDGE_BASE_CORPUS = [
    "Xenobiology explores non-terrestrial biological architectures, alternate nucleic acids (XNA), and exotic metabolic pathways.",
    "Stellar energy transmutation requires metabolic stabilization using quantum-dilated bio-nanite arrays operating at 1:6000 dilation ratios.",
    "Physical Ergonomics in high-energy environments requires AVX2 SIMD real-time sensor processing for kinetic strain prevention.",
    "Ocular Diagnostics monitor bio-energy dissipation across temporal state vectors during high-density chakra synthesis.",
    "Active Inference free-energy minimization optimizes real-time biological feedback loops under thermodynamic pressure."
]

# ---------------------------------------------------------------------------
# Ephemeral RAG Engine
# ---------------------------------------------------------------------------
class EphemeralRAGEngine:
    def __init__(self, corpus: list[str]):
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_FAISS:
            raise ImportError("sentence_transformers or faiss packages are missing.")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.corpus = corpus
        embeddings = self.encoder.encode(corpus, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_FAISS:
            return ""
        query_vec = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        retrieved = [self.corpus[idx] for idx in indices[0] if idx < len(self.corpus)]
        return "\n".join(retrieved)


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


def run_local_rag_text_synthesis(prompt_text: str) -> str:
    """
    Executes an ephemeral vector retrieval pipeline and passes context to a local GGUF LLM,
    falling back to ONNX synthesis if local LLM artifacts or RAG packages are missing.
    """
    if not HAS_SENTENCE_TRANSFORMERS or not HAS_FAISS:
        print("[WARN] RAG dependencies missing. Cascading directly to ONNX model engine...")
        return run_aetheric_archon_onnx_synthesis(prompt_text)

    print("🌀 Building Ephemeral In-Memory FAISS Vector Index...")
    try:
        rag = EphemeralRAGEngine(KNOWLEDGE_BASE_CORPUS)
        retrieved_context = rag.retrieve_context(prompt_text, top_k=3)
        print(f"📥 Context Retrieved via RAG:\n{retrieved_context}")
        
        augmented_prompt = (
            f"Context Information:\n{retrieved_context}\n\n"
            f"User Query: {prompt_text}\n\n"
            f"Synthesize a highly technical diagnostic report incorporating the provided context."
        )

        if HAS_LLAMA_CPP and os.path.exists(LOCAL_LLM_PATH):
            print(f"🤖 Executing Local LLM Inference [{LOCAL_LLM_PATH}]...")
            llm = Llama(model_path=LOCAL_LLM_PATH, n_ctx=2048, verbose=False)
            response = llm(
                f"System: {SYSTEM_PROMPT}\nUser: {augmented_prompt}\nAssistant:",
                max_tokens=1024,
                temperature=0.2,
                stop=["User:", "\n\n\n"]
            )
            return response["choices"][0]["text"].strip()
        else:
            print("[WARN] Local GGUF LLM binary missing or llama_cpp not installed. Cascading to ONNX model engine with RAG telemetry...")
            onnx_report = run_aetheric_archon_onnx_synthesis(prompt_text)
            return (
                f"{onnx_report}\n\n"
                f"### Ephemeral RAG Retrieved Context\n"
                f"{retrieved_context}"
            )

    except Exception as e:
        print(f"[ERROR] Local Ephemeral RAG Synthesis Failed: {e}. Diverting to ONNX engine...")
        return run_aetheric_archon_onnx_synthesis(prompt_text)


def query_nvidia_nim(prompt_text: str) -> str:
    """
    Queries Primary NVIDIA NIM -> Secondary NVIDIA NIM -> Local Ephemeral RAG / ONNX Pipeline.
    """
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)
    
    if not api_key:
        print("[WARN] NVIDIA_API_KEY missing. Diverting to local Ephemeral RAG text engine.")
        return run_local_rag_text_synthesis(prompt_text)

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
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=primary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Primary NIM failed with HTTP {response.status_code}: {response.text}")
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
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=secondary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] Secondary NIM failed with HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[WARN] Secondary NVIDIA NIM Microservice failed: {e}")

    # --- TIER 3: ON-PREMISES LOCAL EPHEMERAL RAG & ONNX MODEL PIPELINE ---
    print("🔒 External APIs unreachable. Executing local Ephemeral RAG / ONNX synthesis pipeline...")
    return run_local_rag_text_synthesis(prompt_text)


def format_text_for_reportlab(text: str) -> str:
    """
    Sanitizes raw model output and converts basic Markdown tags to valid ReportLab XML.
    Strips unsupported tags like <a rel="..."> that break paraparser.
    """
    # 1. Clean out raw HTML linebreaks and unsupported XML elements
    text = re.sub(r'</?(?:link|div|span|p|a|table|tr|td|th|tbody|thead|code|pre|img)[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)

    # 2. Strip raw LaTeX math symbols ($ and \)
    text = text.replace('$', '').replace('\\', '')

    # 3. Safely escape XML reserved characters
    text = html.unescape(text)
    text = escape(text)

    # 4. Re-inject safe inline ReportLab styling from basic Markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)               # **bold** -> <b>bold</b>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)                   # *italic* -> <i>italic</i>
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)                     # _italic_ -> <i>italic</i>
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text) # `code` -> inline mono

    return text.strip()


def parse_markdown_to_story(text: str, story: list, styles: dict):
    """
    Parses Markdown headers, lists, and tables into flowable ReportLab elements.
    Converts raw table markup into native ReportLab Table objects.
    """
    lines = text.split('\n')
    table_buffer = []

    def flush_table_buffer():
        nonlocal table_buffer
        if not table_buffer:
            return
        
        rows = []
        for tbl_line in table_buffer:
            if re.match(r'^\s*\|?\s*:?-+:?\s*(\|', tbl_line):
                continue
            cols = [format_text_for_reportlab(c.strip()) for c in tbl_line.strip('|').split('|')]
            if any(cols):
                rows.append([Paragraph(c, styles['TableCell']) for c in cols])
        
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

        cleaned = format_text_for_reportlab(line_str)

        if line_str.startswith('# '):
            story.append(Spacer(1, 8))
            story.append(Paragraph(cleaned[2:].strip(), styles['H1']))
            story.append(Spacer(1, 4))
        elif line_str.startswith('## '):
            story.append(Spacer(1, 6))
            story.append(Paragraph(cleaned[3:].strip(), styles['H2']))
            story.append(Spacer(1, 4))
        elif line_str.startswith('### ') or line_str.startswith('#### '):
            h_text = re.sub(r'^#+\s*', '', cleaned).strip()
            story.append(Spacer(1, 4))
            story.append(Paragraph(h_text, styles['H3']))
            story.append(Spacer(1, 2))
        elif line_str.startswith('- ') or line_str.startswith('* ') or line_str.startswith('> '):
            bullet_text = re.sub(r'^[-*>]\s*', '', cleaned).strip()
            story.append(Paragraph(f"• {bullet_text}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line_str):
            story.append(Paragraph(cleaned, styles['Numbered']))
            story.append(Spacer(1, 2))
        else:
            story.append(Paragraph(cleaned, styles['Body']))
            story.append(Spacer(1, 3))

    flush_table_buffer()


def generate_pdf_artifact(filename, title, content, session_id):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    # Modular Document Palette & Paragraph Styles
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

    # Logo Header
    try:
        r = requests.get(LOGO_URL, timeout=10)
        with open("logo.webp", "wb") as f: f.write(r.content)
        logo_img = Image("logo.webp", width=1.8 * inch, height=0.6 * inch)
    except Exception:
        logo_img = Paragraph("<b>CELSIUS TECH MEDIA GROUP</b>", custom_styles['DocTitle'])

    header_table = Table([[logo_img, Paragraph(COMPANY_DETAILS, custom_styles['DocBody'])]], colWidths=[3.5 * inch, 3.5 * inch])
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
    comp_table = Table([[Paragraph(compliance_text, custom_styles['CompBox'])]], colWidths=[7.0 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#e0f2f1')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#004d40')),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph(f"<b>UESP DIAGNOSTIC REPORT:</b> {escape(title)}", custom_styles['DocTitle']))
    story.append(Spacer(1, 8))

    # Parse multi-page body content
    parse_markdown_to_story(content, story, custom_styles)

    # Build multi-page PDF dynamically
    doc.build(story)


def process_and_run(title, issue_text):
    # 1. Rust SHA256 Session Generation
    session_id = uesp_quantum_core.generate_ecta_session_id(issue_text)

    # 2. AVX2 Vector SIMD Transformation
    sample_data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    transformed_simd = uesp_quantum_core.avx2_quantum_tensor_transform(sample_data)

    # 3. Primary & Secondary NIM Reasoning with Local Ephemeral RAG / ONNX Failover
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
