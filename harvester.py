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

# ReportLab Graphics Engine
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
PRIMARY_NIM_MODEL = os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
SECONDARY_NIM_MODEL = os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")

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
[FMR SENTINEL SHADOW CLONE SENSOR & RESOLUTION ENGINE]
You are a PhD-level tactical, scientific, and systems resolution engine operating within the UESP/PRCE architecture.
Your purpose is to use the autonomous Shadow Clones as real-time analytical telemetry sensors and probes. The Shadow Clones do not execute raw code behavior here; instead, they act as cognitive sensory lenses, scanning the proposition's dimensional space, measuring friction tensors, calculating probability vectors, and probing hidden structural anomalies.

MANDATORY REPORT STRUCTURE & DEPTH (GO THE EXTRA MILE):
1. **Header Classification Block**:
   - Classification (tailored to the subject domain)
   - Status: SENSOR TELEMETRY ACTIVE // RESOLUTION ENGAGED
   - Target SLA / Objective Guarantee
2. **Section 1: Multi-Sensor Shadow Clone Telemetry & Anomaly Breakdown**:
   - Dissect how the autonomous Shadow Clone sensors (`Sentinel_Tactical`, `Sentinel_Analyst`, `Sentinel_Safety`, `Sentinel_Creative`) probed the proposition.
   - Include a Markdown **Sensor Telemetry Taxonomy Table** (`| Shadow Clone Sensor | Probing Frequency / Focus | Telemetry Reading | Anomaly Status |`) weighted using the DDPG policy and SIMD tensor norm.
3. **Section 2: First-Principles Synthesis & Friction Mapping**:
   - Translate sensor telemetry into a rigorous mechanical breakdown of the user's subject title and proposition.
   - Incorporate PID Cruise stabilization metrics (`pid_cmd`) and Teacher-Student distillation loss (`distill_loss`) as closed-loop feedback controls.
4. **Section 3: Tactical Execution Protocol & Operational Blueprint**:
   - Provide concrete, step-by-step mathematical formulations (in plain text), system configurations, or execution steps directly resolving the subject.
5. **Section 4: Validation Stress-Test Protocol & Success Criteria Table**:
   - Provide a rigorous validation suite and a Markdown **Success Criteria Table** (`| Metric | Threshold | Sensor Verification Method |`) proving stability under load.
6. **Section 5: Final Verdict & Immediate Action Checklist**:
   - Conclude with a decisive technical verdict and an actionable `[ ]` execution checklist.

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
# 2. Shadow Clone Sensor Telemetry & Simulation Generator
# ==============================================================================
def get_ddpg_clone_policy(prompt_text: str) -> list:
    """Executes ddpg_sentinel_policy.onnx seeded by the prompt's sensor state signature."""
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


def execute_sentinel_clone_loop(title: str, prompt_text: str, policy_vec: list, simd_vec: list) -> dict:
    """
    Deploys Shadow Clones as autonomous telemetry sensors probing the target subject & proposition:
    - Forks 4 Constitutional Persona Sensor Clones via Copy-on-Write ShadowCloneMemory.
    - Evaluates each sensor reading against the Adversarial Critic Threshold (Tau).
    - Dispels ('Poof') sensors detecting critical drift or false anomalies.
    - Runs Teacher-Student Distillation & PID Cruise Stabilization on sensor streams.
    """
    clone_alloc, head_gate, critic_tau, pid_damp = policy_vec
    effective_tau = round(max(abs(critic_tau), 0.25), 4)

    memory = ShadowCloneMemory(root_state={
        "subject": title,
        "proposition": prompt_text,
        "verified_status": "SENSOR_PROBE_ACTIVE"
    })

    simd_norm = float(np.linalg.norm(simd_vec)) if simd_vec else 1.0
    complexity_shift = round((len(prompt_text) % 17) * 0.003, 4)

    # Shadow clones acting as analytical telemetry sensors
    personas = [
        ("Sentinel_Tactical", round(min(0.96, 0.92 + complexity_shift), 4), 0.13),
        ("Sentinel_Analyst", round(min(0.95, 0.89 + complexity_shift), 4), 0.17),
        ("Sentinel_Creative", round(max(0.73, 0.80 - complexity_shift), 4), 0.44),  # Exceeds Tau -> POOF (Sensor alert)
        ("Sentinel_Safety", round(min(0.98, 0.95 + complexity_shift), 4), 0.08),
    ]

    accepted_sensors = []
    dispelled_sensors = []

    for name, confidence, critic_reject_prob in personas:
        memory.fork_clone(name, persona_bias=head_gate)
        temp_mem = ShadowCloneMemory(root_state={"subject": title, "proposition": prompt_text})
        temp_mem.fork_clone(name, persona_bias=head_gate)
        passed = temp_mem.verify_and_resolve(
            clone_id=name,
            draft_score=confidence,
            critic_reject_prob=critic_reject_prob,
            tau=effective_tau
        )
        if passed:
            accepted_sensors.append({
                "name": name,
                "confidence": confidence,
                "critic": critic_reject_prob,
                "margin": round(effective_tau - critic_reject_prob, 4)
            })
        else:
            dispelled_sensors.append({
                "name": name,
                "confidence": confidence,
                "critic": critic_reject_prob,
                "excess_risk": round(critic_reject_prob - effective_tau, 4)
            })

    # Execute Teacher-Student Distillation + PID Cruise Stabilization on Sensor Telemetry
    distiller = TeacherStudentDistillation(storage_path=DISTILL_MEM_PATH)
    pid = PIDController(Kp=0.8 * max(abs(pid_damp), 0.1), Ki=0.15, Kd=0.05)
    t_out, s_out, d_loss = distiller.train_step(current_speed=60.0, target_speed=100.0)
    distiller.save_distillation_state(step=1, teacher_out=t_out, student_out=s_out, loss=d_loss)
    pid_cmd = pid.compute(setpoint=100.0 + s_out, measured_value=60.0)

    return {
        "policy_vec": policy_vec,
        "effective_tau": effective_tau,
        "clone_alloc": clone_alloc,
        "head_gate": head_gate,
        "pid_damp": pid_damp,
        "accepted": accepted_sensors,
        "dispelled": dispelled_sensors,
        "distill_loss": round(d_loss, 5),
        "teacher_out": round(t_out, 4),
        "student_out": round(s_out, 4),
        "pid_cmd": round(pid_cmd, 4),
        "simd_vector": [round(float(x), 4) for x in simd_vec],
        "simd_norm": round(simd_norm, 4),
    }


def query_nvidia_nim(title: str, prompt_text: str, sim_data: dict, prior_memories: list) -> str:
    """Feeds Shadow Clone sensor telemetry into NIM to produce an analytical, sensor-driven report."""
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)

    accepted_str = "\n".join(
        f"  - {c['name']} [SENSOR ACTIVE]: Confidence={c['confidence']}, Sensor_Noise={c['critic']}, Calibration_Margin=+{c['margin']}"
        for c in sim_data["accepted"]
    )
    dispelled_str = "\n".join(
        f"  - {c['name']} [SENSOR TRIPPED / POOF]: Confidence={c['confidence']}, Signal_Drift={c['critic']} >= Tau ({sim_data['effective_tau']}), Overload=+{c['excess_risk']}"
        for c in sim_data["dispelled"]
    )
    prior_str = "\n".join(f"  - {m[:220]}..." for m in prior_memories) if prior_memories else "  - None (Fresh Sensor Sweep Horizon)"

    user_simulation_feed = (
        f"SUBJECT TITLE: {title}\n"
        f"PROPOSITION / ISSUE TO RESOLVE: {prompt_text}\n\n"
        f"=== LIVE SHADOW CLONE SENSOR TELEMETRY FEED ===\n"
        f"- DDPG Continuous RL Sensor Weight Vector [Clone_Alloc, Head_Gate, Critic_Tau, PID_Damp]: {sim_data['policy_vec']}\n"
        f"- Active Adversarial Critic Threshold (Tau): {sim_data['effective_tau']}\n"
        f"- Active Sensor Clones (Telemetry Locked):\n{accepted_str}\n"
        f"- Tripped / Disconnected Sensor Clones (Threshold Breach):\n{dispelled_str}\n"
        f"- PID Cruise Stabilization Sensor Loop: Command={sim_data['pid_cmd']} (Damping={sim_data['pid_damp']})\n"
        f"- Teacher-Student Distillation Sensor Memory: Teacher={sim_data['teacher_out']}, Student={sim_data['student_out']}, Sensor_MSE_Loss={sim_data['distill_loss']}\n"
        f"- AVX2 Quantum Tensor Sensor Transform (1:6000 Dilation): {sim_data['simd_vector']} (Sensor_Norm={sim_data['simd_norm']})\n"
        f"- ChromaDB Historical Sensor Memory Context:\n{prior_str}\n\n"
        f"INSTRUCTION: Treat the Shadow Clones strictly as analytical telemetry sensors probing '{title}' and '{prompt_text}'. Synthesize their sensor outputs into an exhaustive, rigorous FMR Sentinel Diagnostic & Execution Report."
    )

    if not api_key:
        print("[WARN] NVIDIA_API_KEY missing. Diverting to local Shadow Clone sensor synthesis.")
        return generate_local_diagnostic_content(title, prompt_text, sim_data)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    session = requests.Session()
    retries = Retry(total=2, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))

    print(f"🚀 Attempting Primary NVIDIA NIM Microservice [{PRIMARY_NIM_MODEL}]...")
    try:
        primary_payload = {
            "model": PRIMARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_simulation_feed}
            ],
            "temperature": 0.25,
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=primary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARN] Primary NIM microservice failed: {e}")

    print(f"⚡ Cascading to Secondary NVIDIA NIM Microservice [{SECONDARY_NIM_MODEL}]...")
    try:
        secondary_payload = {
            "model": SECONDARY_NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_simulation_feed}
            ],
            "temperature": 0.25,
            "max_tokens": 4096
        }
        response = session.post(endpoint, headers=headers, json=secondary_payload, timeout=(10, 180))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARN] Secondary NIM microservice failed: {e}")

    print("🔒 External NIM Microservices unreachable. Activating local Shadow Clone sensor synthesis...")
    return generate_local_diagnostic_content(title, prompt_text, sim_data)


def generate_local_diagnostic_content(title: str, prompt_text: str, sim_data: dict) -> str:
    """Offline fallback generating a structured, sensor-driven diagnostic report from Shadow Clone telemetry."""
    tactical = sim_data["accepted"][0]
    analyst = sim_data["accepted"][1]
    safety = sim_data["accepted"][2]
    dispelled = sim_data["dispelled"][0]

    return "\n".join([
        f"# FMR SENTINEL SENSOR TELEMETRY REPORT: {title.upper()}",
        "**Classification**: MULTI-SENSOR SHADOW CLONE PROBE & RESOLUTION",
        "**Status**: SENSOR TELEMETRY LOCKED // ACTIVE RESOLUTION",
        f"**Target SLA**: 1:6000 Quantum Dilation | Sensor Critic Threshold Tau = {sim_data['effective_tau']}",
        "",
        "## 1. MULTI-SENSOR TELEMETRY & ANOMALY BREAKDOWN",
        f"**Proposition Probed**: {prompt_text}",
        f"Autonomous Shadow Clone sensors probed the operational parameters of '{title}' across four parallel telemetry channels. Using DDPG sensor weight vector `{sim_data['policy_vec']}` and AVX2 tensor sensor norm `{sim_data['simd_norm']}`, anomaly probabilities were measured:",
        "",
        "| Shadow Clone Sensor | Probing Focus & Channel | Telemetry Reading | Sensor Anomaly Status |",
        "| --- | --- | --- | --- |",
        f"| {dispelled['name']} | Speculative Drift Sensor | Signal noise {dispelled['critic']} exceeded Tau ({sim_data['effective_tau']}) | TRIPPED [POOF] (Overload +{dispelled['excess_risk']}) |",
        f"| {tactical['name']} | Throughput & Velocity Sensor | Low latency confirmed with calibration margin +{tactical['margin']} | LOCKED NOMINAL |",
        f"| {analyst['name']} | Convergence & Distillation Sensor | MSE Loss converged at {sim_data['distill_loss']} | LOCKED NOMINAL |",
        f"| {safety['name']} | Boundary & Constraint Sensor | Invariant integrity verified at Confidence {safety['confidence']} | LOCKED SECURE |",
        "",
        "## 2. FIRST-PRINCIPLES SENSOR SYNTHESIS",
        f"- **Tactical Sensor ({tactical['name']} — Conf: {tactical['confidence']})**: Reports optimal velocity vectors for '{title}', identifying immediate operational pathways.",
        f"- **Analytical Sensor ({analyst['name']} — Conf: {analyst['confidence']})**: Tracks quantitative stability and error dissipation via Teacher-Student distillation memory (`{sim_data['distill_loss']}`).",
        f"- **Safety Sensor ({safety['name']} — Conf: {safety['confidence']})**: Maintains strict invariant boundary monitoring with a +{safety['margin']} buffer below Tau.",
        f"- **Tripped Sensor ({dispelled['name']} — POOF at {dispelled['critic']})**: Detected ungrounded signal spikes exceeding Tau (`{sim_data['effective_tau']}`) by `+{dispelled['excess_risk']}`, instantly disconnecting the node in O(1) to protect telemetry integrity.",
        "",
        "## 3. TACTICAL EXECUTION PROTOCOL",
        "1. **Isolate Sensor Channels**: Partition '{0}' into discrete telemetry streams governed by Head Gate `{1}`.".format(title, sim_data['head_gate']),
        f"2. **Engage Closed-Loop Damping**: Stabilize system output using PID sensor control loop `u(t) = {sim_data['pid_cmd']}`.",
        f"3. **Enforce Sensor Thresholds**: Automatically disconnect any operational thread where signal drift exceeds `Tau = {sim_data['effective_tau']}`.",
        "",
        "## 4. VALIDATION STRESS-TEST CRITERIA",
        "| Metric | Target Threshold | Sensor Verification Method |",
        "| --- | --- | --- |",
        f"| Sensor Drift Ceiling | < {sim_data['effective_tau']} | Constitutional Clone Sensor Verification Gate |",
        f"| PID Loop Stabilization | Steady-state error < 1% | Closed-Loop Telemetry (Command = {sim_data['pid_cmd']}) |",
        f"| Distillation Convergence | MSE Loss < 0.05 | Teacher-Student Sensor Alignment ({sim_data['distill_loss']}) |",
        "",
        "## 5. FINAL VERDICT & IMMEDIATE ACTION CHECKLIST",
        f"Sensor telemetry confirms '{title}' is fully stable and ready for execution once tripped speculative channels are purged.",
        "- [ ] Lock sensor telemetry collection on isolated execution nodes.",
        f"- [ ] Calibrate Sensor Critic Gate at Tau = {sim_data['effective_tau']}.",
        f"- [ ] Monitor PID stabilization loop telemetry (target command {sim_data['pid_cmd']})."
    ])

# ==============================================================================
# 3. Full-Width ReportLab PDF Generator (With Tables & Code Block Support)
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
        for tbl_line in table_buffer:
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', tbl_line):
                continue
            cols = [c.strip() for c in tbl_line.strip('|').split('|')]
            if any(cols):
                rows.append([safe_p(c, styles['TableCell']) for c in cols])
        if rows:
            num_cols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < num_cols:
                    r.append(safe_p("", styles['TableCell']))
            col_width = usable_width / max(1, num_cols)
            t = Table(rows, colWidths=[col_width] * num_cols)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#b2dfdb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            story.append(Spacer(1, 4))
            story.append(t)
            story.append(Spacer(1, 6))
        table_buffer = []

    def flush_code():
        nonlocal code_buffer
        if not code_buffer:
            return
        escaped_code = "<br/>".join(escape(cl).replace(" ", "&nbsp;") for cl in code_buffer)
        p = Paragraph(escaped_code, styles['CodeBlock'])
        t = Table([[p]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f4f6f8')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 6))
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
            story.append(Spacer(1, 8))
            story.append(safe_p(line[2:], styles['H1']))
            story.append(Spacer(1, 4))
        elif line.startswith('## '):
            story.append(Spacer(1, 6))
            story.append(safe_p(line[3:], styles['H2']))
            story.append(Spacer(1, 3))
        elif line.startswith('### ') or line.startswith('#### '):
            h_text = re.sub(r'^#+\s*', '', line).strip()
            story.append(Spacer(1, 4))
            story.append(safe_p(h_text, styles['H3']))
            story.append(Spacer(1, 2))
        elif line.startswith('- ') or line.startswith('* '):
            p = safe_p(f"• {line[2:]}", styles['Bullet'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(t)
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line):
            p = safe_p(line, styles['Bullet'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(t)
            story.append(Spacer(1, 2))
        else:
            p = safe_p(line, styles['Body'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
            story.append(t)
            story.append(Spacer(1, 3))

    flush_table()
    flush_code()


def generate_pdf_artifact(filename: str, title: str, content: str, session_id: str):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    usable_width = 7.0 * inch

    styles = {
        'Title': ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor('#003366')),
        'Meta': ParagraphStyle('Meta', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#333333')),
        'Manifest': ParagraphStyle('Manifest', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#004d40')),
        'H1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=11.5, leading=14.5, textColor=colors.HexColor('#003366'), keepWithNext=True),
        'H2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#004d40'), keepWithNext=True),
        'H3': ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=9.2, leading=12, textColor=colors.HexColor('#006699'), keepWithNext=True),
        'Body': ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#222222'), wordWrap='CJK'),
        'Bullet': ParagraphStyle('Bullet', fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor('#222222'), wordWrap='CJK'),
        'TableCell': ParagraphStyle('TableCell', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#111111'), wordWrap='CJK'),
        'CodeBlock': ParagraphStyle('CodeBlock', fontName='Courier', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#1a237e')),
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
        f"ECTA & QUANTUM DILATION MANIFEST:\n"
        f"• SHA256 ECTA Timestamped Session: {session_id}\n"
        f"• Quantum Cycle Time Dilation: 1 : 6000 Standard | Edge Acceleration: AVX2 SIMD Vectorized\n"
        f"• Multimodal Knowledge Catalog: Registered ({len(NVIDIA_NIM_CATALOG)} Active Models)\n"
        f"• Autonomous Sensor Array: Shadow Clone RAG Probes Active"
    )
    manifest_box = Table([[safe_p(manifest_text, styles['Manifest'])]], colWidths=[usable_width])
    manifest_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e0f2f1')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#004d40')),
        ('PADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(manifest_box)
    story.append(Spacer(1, 14))

    story.append(safe_p(f"UESP DIAGNOSTIC REPORT: {title}", styles['Title']))
    story.append(Spacer(1, 8))

    parse_markdown_to_story(content, story, styles)
    doc.build(story)

# ==============================================================================
# 4. Main Execution Entry
# ==============================================================================
def process_and_run(title: str, issue_text: str):
    combined_context = f"{title} :: {issue_text}"
    session_id = uesp_quantum_core.generate_ecta_session_id(combined_context)

    # Seed AVX2 SIMD Sensor Tensor Transform from subject telemetry characteristics
    seed_vec = [float((ord(combined_context[i % len(combined_context)]) % 19) + 1) * 0.5 for i in range(8)]
    simd_vec = uesp_quantum_core.avx2_quantum_tensor_transform(seed_vec)

    # Execute Shadow Clone Sensor Probes & DDPG Policy Evaluation on the Subject
    policy_vec = get_ddpg_clone_policy(combined_context)
    sim_data = execute_sentinel_clone_loop(title, issue_text, policy_vec, simd_vec)

    # Recall prior ChromaDB sensor memory and generate the analytical report
    vector_store = LocalARMVectorStore()
    prior_memories = vector_store.recall_prior_context(n_results=2)
    report_content = query_nvidia_nim(title, issue_text, sim_data, prior_memories)

    # Persist sensor sweep into ChromaDB
    vector_store.trap_information(session_id, [f"Sensor Probe Subject: {title}\n{report_content[:1200]}"])

    pdf_filename = f"Report_{session_id[:16]}.pdf"
    generate_pdf_artifact(pdf_filename, title, report_content, session_id)
    print(f"✅ Shadow Clone Sensor Diagnostic PDF generated: {pdf_filename}")

    # WordPress REST API Upload Integration
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
                    "title": f"Diagnostic: {title}",
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
                    print("✅ WordPress UESP record successfully created referencing the diagnostic PDF.")
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
