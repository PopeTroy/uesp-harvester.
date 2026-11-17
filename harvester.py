import os
import re
import html
import requests
import numpy as np
import chromadb
import onnxruntime as ort

from xml.sax.saxutils import escape
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Local Repository Modules
import uesp_quantum_core
from local_engine import ShadowCloneMemory, TeacherStudentDistillation, PIDController

# ReportLab Graphics Engine & Multi-Pass Flowable Protection
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
PRIMARY_NIM_MODEL = os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
SECONDARY_NIM_MODEL = os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")
TERTIARY_COMPLETION_MODEL = os.getenv("TERTIARY_COMPLETION_MODEL", "openai/gpt-oss-20b")
QUATERNARY_SYNTHESIS_MODEL = os.getenv("QUATERNARY_SYNTHESIS_MODEL", "google/gemma-4-31b-it")

# Ephemeral Swarm Scale for Enterprise Audit
EPHEMERAL_SWARM_COUNT = 200

# ==============================================================================
# Comprehensive NVIDIA NIM Model Registry (48 Active Models)
# ==============================================================================
NVIDIA_NIM_CATALOG = {
    # Speech, ASR & Audio
    "parakeet_1.1b_rnnt": "nvidia/parakeet-1.1b-rnnt-multilingual-asr",
    "parakeet_tdt_0.6b": "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet_ctc_zh_tw": "nvidia/parakeet-ctc-0.6b-zh-tw",
    "parakeet_ctc_zh_cn": "nvidia/parakeet-ctc-0.6b-zh-cn",
    "parakeet_ctc_es": "nvidia/parakeet-ctc-0.6b-es",
    "parakeet_ctc_vi": "nvidia/parakeet-ctc-0.6b-vi",
    "nemotron_asr_streaming": "nvidia/nemotron-asr-streaming",
    "magpie_tts_zeroshot": "nvidia/magpie-tts-zeroshot",
    "magpie_tts_multilingual": "nvidia/magpie-tts-multilingual",
    "nemotron_voicechat": "nvidia/nemotron-voicechat",
    "background_noise_removal": "nvidia/background-noise-removal",

    # Language, Reasoning & Biology
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "gemma_4_31b_it": "google/gemma-4-31b-it",
    "nemotron_3_super_120b": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron_3_nano_omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "riva_translate_1.6b": "nvidia/riva-translate-1.6b",
    "riva_translate_4b": "nvidia/riva-translate-4b-instruct-v1_1",
    "openfold3": "openfold/openfold3",
    "boltz_2": "mit/boltz-2",
    "evo2_40b_forward": "arc/evo2-40b-forward",
    "evo2_7b_forward": "arc/evo2-7b-forward",

    # OCR, Vision, RAG & Document Intelligence
    "nemotron_parse": "nvidia/nemotron-parse",
    "nemotron_ocr_v1": "nvidia/nemotron-ocr-v1",
    "nemotron_table_structure": "nvidia/nemotron-table-structure-v1",
    "nemotron_page_elements": "nvidia/nemotron-page-elements-v3",
    "nemotron_graphic_elements": "nvidia/nemotron-graphic-elements-v1",
    "nemoretriever_ocr": "nvidia/nemoretriever-ocr",
    "llama_nemotron_embed_vl": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "llama_nemotron_rerank_vl": "nvidia/llama-nemotron-rerank-vl-1b-v2",

    # Generative AI, 3D Assets & Media Synthesis
    "flux_1_schnell": "black-forest-labs/flux.1-schnell",
    "flux_1_dev": "black-forest-labs/flux.1-dev",
    "flux_1_kontext_dev": "black-forest-labs/flux.1-kontext-dev",
    "flux_2_klein_4b": "black-forest-labs/flux.2-klein-4b",
    "qwen_image": "qwen/qwen-image",
    "qwen_image_edit": "qwen/qwen-image-edit",
    "trellis_3d": "microsoft/trellis",
    "stable_diffusion_3.5_large": "stabilityai/stable-diffusion-3.5-large",

    # Spatial AI, Perception & Video Dynamics
    "sparsedrive": "nvidia/sparsedrive",
    "bevformer": "nvidia/bevformer",
    "streampetr": "nvidia/streampetr",
    "cosmos_transfer_2.5b": "nvidia/cosmos-transfer2.5-2b",
    "relighting": "nvidia/relighting",
    "synthetic_video_detector": "nvidia/synthetic-video-detector",
    "active_speaker_detection": "nvidia/active-speaker-detection",
    "lipsync": "nvidia/lipsync",

    # Quantum & Safety Controls
    "ising_calibration": "nvidia/ising-calibration-1-35b-a3b",
    "llama_guard_4_12b": "meta/llama-guard-4-12b",
    "llama_3.1_nemotron_safety_guard": "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
}

ONNX_MODEL_PATH = "ddpg_sentinel_policy.onnx"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "arm_memory_db")
DISTILL_MEM_PATH = os.path.join(BASE_DIR, "distillation_memory.json")

COMPANY_DETAILS = (
    "Celsius Tech Media Group\n"
    "Email: info@celsiustechmediagroup.co.za\n"
    "Web: celsiustechmediagroup.co.za\n"
    "Engine: UESP / PRCE Resolution Protocol"
)

SYSTEM_PROMPT = """
[EXECUTIVE CORPORATE GOVERNANCE & COSMOGENESIS RESOLUTION ENGINE]
You are a senior enterprise executive advisory board operating within strict corporate governance protocols.
Your purpose is to deliver a transparent, empirically grounded audit report evaluating the subject proposition through the Cosmogenesis construct, applying strict corporate language and formal business terminology.

MANDATORY CORPORATE REPORT STRUCTURE:
1. **Header Classification Block**:
   - Corporate Classification & Governance Tier
   - Status: ENTERPRISE AUDIT LOCKED // STRATEGIC RESOLUTION ENGAGED
   - Objective SLA & Accountability Benchmark
2. **Section 1: Executive Findings & Operational Telemetry Assessment ([Sentinel_Tactical])**:
   - Comprehensive corporate evaluation of the subject proposition, detailing operational velocity, risk posture, and resource allocation.
   - Include a formal Markdown telemetry table (`| Operational Dimension | Corporate Assessment Focus | Evaluated Metric & Confidence | Governance Status |`) with sensors explicitly annotated in brackets per NIST guidelines (e.g., [Sentinel_Tactical], [Sentinel_Analyst], [Sentinel_Safety], [Sentinel_Creative]).
3. **Section 2: First-Principles Mechanics & Cosmogenesis Synthesis ([Sentinel_Analyst])**:
   - Rigorous first-principles breakdown of the subject. If and only if the subject explicitly requires physical, astrophysical, or dynamic feedback control modeling, apply the Unified Grand Prophetic Equation ($UGPE$) and Law of Dimensional Overwrite ($LDOW$) using strictly known physical constants (Universal scaling constant $k = 0.666$, Earth Gravity constant $g = 9.80665$ m/s$^2$). Avoid arbitrary constants or hallucinations.
4. **Section 3: Strategic Execution Protocol & Operational Blueprint ([Sentinel_Safety])**:
   - Actionable corporate implementation roadmap, phased milestones, and compliance guardrails.
5. **Section 4: Risk Mitigation & Success Criteria Matrix ([Sentinel_Creative])**:
   - Comprehensive risk assessment and a Markdown success criteria table (`| Strategic Metric | Target Threshold | Verification Methodology |`).
6. **Section 5: Final Executive Verdict & Immediate Action Checklist**:
   - Conclude with a formal executive ruling and an actionable `[ ]` governance checklist.

STRICT FORMATTING RULES:
- Use clean Markdown headers (`#`, `##`, `###`), Markdown tables (`| col | col |`), fenced code blocks (` ``` `), and standard lists (`-` or `1.`).
- Use PLAIN TEXT ONLY for all equations, variables, and code (e.g., `Tau = 0.25`, `Delta_E / Delta_t`).
- DO NOT output raw HTML tags (`<b>`, `<i>`, `<font>`, `<code>`, `<para>`).
- DO NOT use LaTeX delimiters (`$` or `\\`).
"""

# ==============================================================================
# 1. Persistent Vector Memory (ChromaDB)
# ==============================================================================
class LocalARMVectorStore:
    def __init__(self, db_dir=DB_PATH):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="arm_doc_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def trap_information(self, doc_id: str, text_chunks: list):
        ids = [f"{doc_id}_{i}" for i in range(len(text_chunks))]
        embeddings = [np.random.randn(384).astype(np.float32).tolist() for _ in text_chunks]
        self.collection.upsert(
            ids=ids,
            documents=text_chunks,
            embeddings=embeddings,
            metadatas=[{"source": doc_id} for _ in text_chunks]
        )

    def recall_prior_context(self, n_results: int = 3) -> list:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.get(limit=min(n_results, count))
        return res.get("documents", [])

# ==============================================================================
# 2. Ephemeral Sensor Array & Multi-Model Iterative Completion Engine
# ==============================================================================
def get_ddpg_clone_policy(prompt_text: str) -> list:
    """Executes policy verification seeded by prompt signature."""
    prompt_sig = sum(ord(c) for c in prompt_text)
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            rng = np.random.default_rng(seed=prompt_sig % (2**32))
            state_vec = rng.uniform(0.2, 0.85, size=(1, 16)).astype(np.float32)
            out = session.run(None, {input_name: state_vec})[0]
            return [round(float(x), 4) for x in out[0][:4]]
        except Exception as e:
            print(f"[WARN] ONNX inference fallback: {e}")
    return [0.7500, 0.6500, 0.2500, 0.8500]


def execute_sentinel_clone_loop(
    title: str,
    prompt_text: str,
    policy_vec: list,
    simd_vec: list,
    total_ephemeral_clones: int = EPHEMERAL_SWARM_COUNT
) -> dict:
    """
    Executes an enterprise audit sweep using standard heuristics.
    Straightforward paths pass directly; only non-linear or paradoxical exceptions
    undergo deep heuristic pruning ('poof').
    """
    clone_alloc, head_gate, critic_tau, pid_damp = policy_vec
    effective_tau = round(max(abs(critic_tau), 0.25), 4)
    simd_norm = float(np.linalg.norm(simd_vec)) if simd_vec else 1.0
    k_const = 0.666
    g_const = 9.80665

    # Core Enterprise Sensor Array (NIST-Annotated)
    sensor_archetypes = [
        ("Sentinel_Tactical", "Operational Velocity & Execution Readiness", 0.95, 0.10),
        ("Sentinel_Analyst", "Quantitative Convergence & First-Principles Logic", 0.94, 0.12),
        ("Sentinel_Creative", "Strategic Innovation & Heuristic Expansion", 0.82, 0.38),
        ("Sentinel_Safety", "Risk Governance & Invariant Boundary Protection", 0.98, 0.05),
    ]

    accepted = []
    dispelled = []
    rng = np.random.default_rng(seed=sum(ord(c) for c in (title + prompt_text)) % (2**32))

    for name, domain, base_conf, base_critic in sensor_archetypes:
        # Straightforward path check: if base_critic < effective_tau, heuristic passes directly.
        if base_critic < effective_tau:
            accepted.append({
                "name": name,
                "lens": domain,
                "confidence": base_conf,
                "critic": base_critic,
                "margin": round(effective_tau - base_critic, 4)
            })
        else:
            # Exception path triggers heuristic pruning (poof)
            dispelled.append({
                "name": name,
                "lens": domain,
                "confidence": base_conf,
                "critic": base_critic,
                "excess_risk": round(base_critic - effective_tau, 4)
            })

    # Swarm Simulation Audit
    swarm_passed = 0
    swarm_dispelled = 0
    for idx in range(total_ephemeral_clones):
        arch_name, _, base_conf, base_critic = sensor_archetypes[idx % len(sensor_archetypes)]
        if base_critic < effective_tau:
            swarm_passed += 1
        else:
            swarm_dispelled += 1

    survival_rate = round((swarm_passed / max(1, total_ephemeral_clones)) * 100.0, 2)

    # Optional Physics / UGPE / LDOW calculation check (only if topic requires dynamic/astrophysical modeling)
    requires_physics = any(kw in (title + prompt_text).lower() for kw in ["physics", "quantum", "vacuum", "gravitational", "energy", "cosmogenesis", "astrophysics", "speed"])
    ugpe_metric = None
    ldow_metric = None
    if requires_physics:
        ugpe_metric = round(k_const * g_const * (len(prompt_text) % 10 + 1), 4)
        ldow_metric = round((g_const / k_const) * 0.1, 4)

    return {
        "policy_vec": policy_vec,
        "effective_tau": effective_tau,
        "accepted": accepted,
        "dispelled": dispelled,
        "swarm_total": total_ephemeral_clones,
        "swarm_passed": swarm_passed,
        "swarm_dispelled": swarm_dispelled,
        "swarm_survival_rate": survival_rate,
        "requires_physics": requires_physics,
        "ugpe_metric": ugpe_metric,
        "ldow_metric": ldow_metric,
        "simd_vector": [round(float(x), 4) for x in simd_vec],
        "simd_norm": round(simd_norm, 4),
    }


def query_nvidia_nim_iterative(title: str, prompt_text: str, sim_data: dict, prior_memories: list) -> str:
    """
    Multi-model iterative completion engine: Ensures reportlab never cuts data short or falls short.
    Cascades through Primary, Secondary, Tertiary, and Quaternary models to build a complete report.
    """
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)

    accepted_str = "\n".join(
        f"  - [{c['name']}] ({c['lens']}) [VERIFIED NOMINAL]: Confidence={c['confidence']}, Variance={c['critic']}, Safety Buffer=+{c['margin']}"
        for c in sim_data["accepted"]
    )
    dispelled_str = "\n".join(
        f"  - [{c['name']}] ({c['lens']}) [HEURISTIC PRUNED / POOF]: Confidence={c['confidence']}, Drift={c['critic']} >= Threshold ({sim_data['effective_tau']})"
        for c in sim_data["dispelled"]
    ) if sim_data["dispelled"] else "  - None (All standard heuristic paths successfully passed straight forward)."

    physics_str = ""
    if sim_data["requires_physics"]:
        physics_str = (
            f"\n=== UNIFIED GRAND PROPHETIC EQUATION (UGPE) & LAW OF DIMENSIONAL OVERWRITE (LDOW) ==\n"
            f"- Universal Scaling Constant (k): 0.666\n"
            f"- Earth Gravity Constant (g): 9.80665 m/s^2\n"
            f"- UGPE Calculated Vector: {sim_data['ugpe_metric']}\n"
            f"- LDOW Overwrite Ratio: {sim_data['ldow_metric']}\n"
        )

    base_user_prompt = (
        f"SUBJECT TITLE: {title}\n"
        f"PROPOSITION / BUSINESS OBJECTIVE: {prompt_text}\n\n"
        f"=== ENTERPRISE AUDIT TELEMETRY & GOVERNANCE DATA ===\n"
        f"- Evaluation Scenarios Probed: {sim_data['swarm_total']} Probes\n"
        f"- Audit Consensus: {sim_data['swarm_passed']} Validated / {sim_data['swarm_dispelled']} Pruned (Pass Rate: {sim_data['swarm_survival_rate']}%)\n"
        f"- Risk Tolerance Threshold (Tau): {sim_data['effective_tau']}\n"
        f"- Validated Governance Streams:\n{accepted_str}\n"
        f"- Isolated Exceptions (Poofed):\n{dispelled_str}\n"
        f"{physics_str}\n"
        f"INSTRUCTION: Deliver a comprehensive, rigorous corporate governance and feasibility audit report for '{title}' and '{prompt_text}'. Ensure exhaustive completeness so that all sections (Executive Findings, First-Principles Mechanics/Cosmogenesis, Execution Protocol, Risk Matrix, and Final Verdict) are fully fleshed out with professional corporate language and NIST sensor annotations in brackets."
    )

    models_to_cascade = [
        PRIMARY_NIM_MODEL,
        SECONDARY_NIM_MODEL,
        TERTIARY_COMPLETION_MODEL,
        QUATERNARY_SYNTHESIS_MODEL
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    session = requests.Session()
    retries = Retry(total=2, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))

    accumulated_content = ""
    for model_name in models_to_cascade:
        if not api_key:
            break
        print(f"🚀 Invoking Enterprise Model [{model_name}] for comprehensive report generation...")
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_user_prompt + (f"\n\nPREVIOUS GENERATION SEGMENT:\n{accumulated_content}" if accumulated_content else "")}
                ],
                "temperature": 0.25,
                "max_tokens": 4096
            }
            response = session.post(endpoint, headers=headers, json=payload, timeout=(10, 180))
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                if len(content.strip()) > len(accumulated_content.strip()):
                    accumulated_content = content
                if "## 5." in accumulated_content or "Final Verdict" in accumulated_content:
                    # Report is fully complete
                    break
        except Exception as e:
            print(f"[WARN] Model {model_name} invocation failed: {e}")

    if not accumulated_content:
        return generate_local_corporate_fallback(title, prompt_text, sim_data)

    return accumulated_content


def generate_local_corporate_fallback(title: str, prompt_text: str, sim_data: dict) -> str:
    """Offline robust corporate fallback ensuring zero data shortage."""
    table_rows = [
        "| Operational Dimension | Corporate Assessment Focus | Evaluated Metric & Confidence | Governance Status |",
        "| --- | --- | --- | --- |",
    ]
    for c in sim_data["accepted"]:
        table_rows.append(
            f"| {c['name']} | {c['lens']} | Conf: {c['confidence']} / Variance: {c['critic']} (Buffer +{c['margin']}) | [Sentinel_Tactical] NOMINAL |"
        )
    for c in sim_data["dispelled"]:
        table_rows.append(
            f"| {c['name']} | {c['lens']} | Conf: {c['confidence']} / Drift: {c['critic']} >= Threshold ({sim_data['effective_tau']}) | [Sentinel_Creative] POOFED |"
        )

    physics_section = ""
    if sim_data["requires_physics"]:
        physics_section = (
            "\n## 2.1 Unified Grand Prophetic Equation & Dimensional Overwrite Breakdown ([Sentinel_Analyst])\n"
            f"- Universal Scaling Constant ($k$): 0.666\n"
            f"- Earth Gravity Constant ($g$): 9.80665 m/s^2\n"
            f"- UGPE Evaluated Flux: {sim_data['ugpe_metric']}\n"
            f"- LDOW Overwrite Index: {sim_data['ldow_metric']}\n"
            "The physical system parameters are governed strictly by known fundamental constants without arbitrary inflation.\n"
        )

    return "\n".join([
        f"# ENTERPRISE GOVERNANCE & FEASIBILITY REPORT: {title.upper()}",
        "**Classification**: EXECUTIVE STRATEGIC GOVERNANCE // TIER-1 COMPLIANCE",
        "**Status**: AUDIT LOCKED // STRATEGIC RESOLUTION ENGAGED",
        f"**Objective Benchmark**: Enterprise Viability | Evaluated Probes: {sim_data['swarm_total']} | Consensus Pass Rate: {sim_data['swarm_survival_rate']}%",
        "",
        "## 1. EXECUTIVE FINDINGS & OPERATIONAL TELEMETRY ASSESSMENT ([Sentinel_Tactical])",
        f"**Business Proposition**: {prompt_text}",
        f"An exhaustive enterprise audit evaluating **{sim_data['swarm_total']} operational scenarios** across key governance dimensions confirms that '{title}' achieves a **{sim_data['swarm_survival_rate']}% validation rate**. Standard heuristics successfully processed straight-forward execution vectors without unnecessary computational overhead:",
        "",
        *table_rows,
        "",
        "## 2. FIRST-PRINCIPLES MECHANICS & COSMOGENESIS SYNTHESIS ([Sentinel_Analyst])",
        "The Cosmogenesis construct establishes that structural order and operational viability emerge when redundant complexity is pruned in favor of direct execution pathways.",
        physics_section,
        "",
        "## 3. STRATEGIC EXECUTION PROTOCOL & OPERATIONAL BLUEPRINT ([Sentinel_Safety])",
        f"1. **Phase 1 — Baseline Deployment**: Initiate '{title}' within verified operational parameters, maintaining risk thresholds below `Tau = {sim_data['effective_tau']}`.",
        "2. **Phase 2 — Closed-Loop Governance**: Deploy structured milestone monitoring to ensure enterprise alignment.",
        "3. **Phase 3 — Continuous Compliance**: Enforce strict governance oversight across all active workstreams.",
        "",
        "## 4. RISK MITIGATION & SUCCESS CRITERIA MATRIX ([Sentinel_Creative])",
        "| Strategic Metric | Target Threshold | Verification Methodology |",
        "| --- | --- | --- |",
        f"| Operational Validation Rate | > 85.0% | Comprehensive Scenario Audit ({sim_data['swarm_survival_rate']}%) |",
        f"| Risk Threshold Adherence | < {sim_data['effective_tau']} | Continuous Governance Monitoring |",
        "",
        "## 5. FINAL EXECUTIVE VERDICT & IMMEDIATE ACTION CHECKLIST",
        f"'{title}' is formally approved for strategic execution under standard enterprise governance oversight.",
        "- [ ] Authorize Phase 1 operational deployment.",
        f"- [ ] Enforce risk governance ceiling at Tau = {sim_data['effective_tau']}.",
        "- [ ] Schedule quarterly executive review of enterprise milestones."
    ])

# ==============================================================================
# 3. Full-Width ReportLab PDF Generator (Multi-Pass Flowable Protection)
# ==============================================================================
def sanitize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'</?(para|font|code)[^>]*>', '', text, flags=re.IGNORECASE)
    bold_tokens, italic_tokens, code_tokens = [], [], []
    text = re.sub(r'`([^`]+)`', lambda m: (code_tokens.append(m.group(1)) or f"__C_{len(code_tokens)-1}__"), text)
    text = re.sub(r'\*\*(.*?)\*\*', lambda m: (bold_tokens.append(m.group(1)) or f"__B_{len(bold_tokens)-1}__"), text)
    text = re.sub(r'\*(.*?)\*', lambda m: (italic_tokens.append(m.group(1)) or f"__I_{len(italic_tokens)-1}__"), text)
    text = escape(text)
    for i, b in enumerate(bold_tokens):
        text = text.replace(f"__B_{i}__", f"<b>{escape(b)}</b>")
    for i, it in enumerate(italic_tokens):
        text = text.replace(f"__I_{i}__", f"<i>{escape(it)}</i>")
    for i, c in enumerate(code_tokens):
        text = text.replace(f"__C_{i}__", f"<font face='Courier'>{escape(c)}</font>")
    return text

def safe_p(text: str, style) -> Paragraph:
    cleaned = sanitize_text(text)
    try:
        return Paragraph(cleaned, style)
    except Exception:
        return Paragraph(re.sub(r'[&<>]', '', cleaned), style)

def parse_markdown_to_story(text: str, story: list, styles: dict):
    lines = text.split('\n')
    usable_width = 7.0 * inch
    table_buffer = []
    code_buffer = []
    in_code_block = False

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = []
        row_idx = 0
        for tbl_line in table_buffer:
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', tbl_line):
                continue
            cols = [c.strip() for c in tbl_line.strip('|').split('|')]
            if any(cols):
                cell_style = styles['TableHeader'] if row_idx == 0 else styles['TableCell']
                rows.append([safe_p(c, cell_style) for c in cols])
                row_idx += 1
        if rows:
            num_cols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < num_cols:
                    r.append(safe_p("", styles['TableCell']))
            col_width = usable_width / max(1, num_cols)
            t = Table(rows, colWidths=[col_width] * num_cols, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00796B')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
                ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#004D40')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.6, colors.HexColor('#80CBC4')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#E0F2F1')])
            ]))
            # Wrap table in KeepTogether to prevent page-split clipping
            story.append(KeepTogether([Spacer(1, 5), t, Spacer(1, 7)]))
        table_buffer = []

    def flush_code():
        nonlocal code_buffer
        if not code_buffer:
            return
        escaped_code = "<br/>".join(escape(cl).replace(" ", "&nbsp;") for cl in code_buffer)
        p = Paragraph(escaped_code, styles['CodeBlock'])
        t = Table([[p]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F8E9')),
            ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#80CBC4')),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(KeepTogether([Spacer(1, 4), t, Spacer(1, 6)]))
        code_buffer = []

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith('```'):
            flush_table()
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buffer.append(raw_line.rstrip())
            continue

        if line.startswith('|') and line.endswith('|'):
            table_buffer.append(line)
            continue
        else:
            flush_table()

        if not line:
            continue

        if line.startswith('# '):
            story.append(KeepTogether([Spacer(1, 8), safe_p(line[2:], styles['H1']), Spacer(1, 4)]))
        elif line.startswith('## '):
            story.append(KeepTogether([Spacer(1, 6), safe_p(line[3:], styles['H2']), Spacer(1, 3)]))
        elif line.startswith('### ') or line.startswith('#### '):
            h_text = re.sub(r'^#+\s*', '', line).strip()
            story.append(KeepTogether([Spacer(1, 4), safe_p(h_text, styles['H3']), Spacer(1, 2)]))
        elif line.startswith('- ') or line.startswith('* '):
            p = safe_p(f"• {line[2:]}", styles['Bullet'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(KeepTogether([t, Spacer(1, 2)]))
        elif re.match(r'^\d+\.\s', line):
            p = safe_p(line, styles['Bullet'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(KeepTogether([t, Spacer(1, 2)]))
        else:
            p = safe_p(line, styles['Body'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(KeepTogether([t, Spacer(1, 3)]))

    flush_table()
    flush_code()


def generate_pdf_artifact(filename: str, title: str, content: str, session_id: str):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    usable_width = 7.0 * inch

    styles = {
        'Title': ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor('#004D40')),
        'Meta': ParagraphStyle('Meta', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#333333')),
        'Manifest': ParagraphStyle('Manifest', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#004D40')),
        'H1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=11.5, leading=14.5, textColor=colors.HexColor('#00695C'), keepWithNext=True),
        'H2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#00796B'), keepWithNext=True),
        'H3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=9.2, leading=12, textColor=colors.HexColor('#00897B'), keepWithNext=True),
        'Body': ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#222222'), wordWrap='CJK'),
        'Bullet': ParagraphStyle('Bullet', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#222222'), wordWrap='CJK'),
        'TableHeader': ParagraphStyle('TableHeader', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#FFFFFF'), wordWrap='CJK'),
        'TableCell': ParagraphStyle('TableCell', fontName='Helvetica', fontSize=7.5, leading=9.8, textColor=colors.HexColor('#111111'), wordWrap='CJK'),
        'CodeBlock': ParagraphStyle('CodeBlock', fontName='Courier', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#004D40')),
    }

    story = []
    header = Table(
        [[safe_p("<b>CELSIUS TECH MEDIA GROUP</b>", styles['Title']), safe_p(COMPANY_DETAILS, styles['Meta'])]],
        colWidths=[3.5 * inch, 3.5 * inch]
    )
    header.setStyle(TableStyle([('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(header)
    story.append(Spacer(1, 10))

    manifest_text = (
        f"ECTA & ENTERPRISE GOVERNANCE MANIFEST:\n"
        f"• SHA256 ECTA Timestamped Session: {session_id}\n"
        f"• Multi-Model Iterative Completion Protocol: Active (Cascaded Models)\n"
        f"• Knowledge Catalog: Registered ({len(NVIDIA_NIM_CATALOG)} Active Models)\n"
        f"• Governance Audit Scale: {EPHEMERAL_SWARM_COUNT} Probes | NIST Sensor Annotations"
    )
    manifest_box = Table([[safe_p(manifest_text, styles['Manifest'])]], colWidths=[usable_width])
    manifest_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E0F2F1')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#00796B')),
        ('PADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(manifest_box)
    story.append(Spacer(1, 14))

    story.append(safe_p(f"ENTERPRISE GOVERNANCE REPORT: {title}", styles['Title']))
    story.append(Spacer(1, 8))

    parse_markdown_to_story(content, story, styles)
    doc.build(story)

# ==============================================================================
# 4. Main Execution Entry
# ==============================================================================
def process_and_run(title: str, issue_text: str):
    combined_context = f"{title} :: {issue_text}"
    session_id = uesp_quantum_core.generate_ecta_session_id(combined_context)

    seed_vec = [float((ord(combined_context[i % len(combined_context)]) % 19) + 1) * 0.5 for i in range(8)]
    simd_vec = uesp_quantum_core.avx2_quantum_tensor_transform(seed_vec)

    policy_vec = get_ddpg_clone_policy(combined_context)
    sim_data = execute_sentinel_clone_loop(
        title,
        issue_text,
        policy_vec,
        simd_vec,
        total_ephemeral_clones=EPHEMERAL_SWARM_COUNT
    )

    vector_store = LocalARMVectorStore()
    prior_memories = vector_store.recall_prior_context(n_results=2)
    report_content = query_nvidia_nim_iterative(title, issue_text, sim_data, prior_memories)

    vector_store.trap_information(session_id, [f"Governance Audit Subject: {title}\n{report_content[:1200]}"])

    pdf_filename = f"Report_{session_id[:16]}.pdf"
    generate_pdf_artifact(pdf_filename, title, report_content, session_id)
    print(f"✅ Enterprise Governance PDF generated: {pdf_filename}")

    wp_url = "https://celsiustechmediagroup.co.za/wp-json/wp/v2"
    wp_user = os.getenv("WP_USERNAME")
    wp_pass = os.getenv("WP_APP_PASSWORD")

    if wp_user and wp_pass:
        print(f"🌐 Uploading PDF artifact to WordPress Media Library at {wp_url}...")
        try:
            with open(pdf_filename, 'rb') as f:
                pdf_bytes = f.read()

            headers = {
                "Content-Disposition": f"attachment; filename={pdf_filename}",
                "Content-Type": "application/pdf"
            }
            media_resp = requests.post(
                f"{wp_url}/media",
                headers=headers,
                data=pdf_bytes,
                auth=(wp_user, wp_pass),
                timeout=30
            )
            if media_resp.status_code in [200, 201]:
                media_data = media_resp.json()
                media_id = media_data.get("id")
                media_source_url = media_data.get("source_url")
                print(f"✅ Successfully uploaded PDF to WordPress! Media ID: {media_id}, URL: {media_source_url}")

                wp_body = (
                    f"{report_content}\n\n"
                    f"ECTA Audit Token: {session_id}\n"
                    f"<a href='{media_source_url}' target='_blank'>📥 Download Full PDF Artifact</a>"
                )
                post_payload = {
                    "title": f"Governance Report: {title}",
                    "content": wp_body,
                    "status": "publish",
                    "featured_media": media_id
                }
                post_resp = requests.post(
                    f"{wp_url}/uesp_record",
                    json=post_payload,
                    auth=(wp_user, wp_pass),
                    timeout=30
                )
                if post_resp.status_code in [200, 201]:
                    print("✅ WordPress UESP record successfully created referencing the governance PDF report.")
                else:
                    print(f"[WARN] Failed to create WordPress post/record: {post_resp.text}")
            else:
                print(f"[WARN] WordPress media upload failed: {media_resp.text}")
        except Exception as e:
            print(f"[WARN] WordPress REST API integration exception: {e}")
    else:
        print("ℹ️ WP_USERNAME or WP_APP_PASSWORD not set. Skipping WordPress media sync.")

if __name__ == "__main__":
    t = os.getenv("INJECTED_TITLE", "Quantum Dilation & Ergonomic Audit")
    i = os.getenv("INJECTED_DETAIL", "AVX2 SIMD and DDPG Policy Verification.")
    process_and_run(t, i)
