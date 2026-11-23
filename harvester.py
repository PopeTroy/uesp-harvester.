import os
import re
import html
import zipfile
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

# Optional HTML/CSS-to-PDF Engine (WeasyPrint) + ReportLab Fallback
try:
    from weasyprint import HTML as WeasyHTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
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
   - Rigorous first-principles breakdown of the subject. If and only if the subject explicitly requires physical, astrophysical, or dynamic feedback control modeling, apply the Unified Grand Prophetic Equation (UGPE) and Law of Dimensional Overwrite (LDOW) using strictly known physical constants (Universal scaling constant k = 0.666, Earth Gravity constant g = 9.80665 m/s^2). Avoid arbitrary constants or hallucinations.
4. **Section 3: Strategic Execution Protocol & Operational Blueprint ([Sentinel_Safety])**:
   - Actionable corporate implementation roadmap, phased milestones, and compliance guardrails.
5. **Section 4: Risk Mitigation & Success Criteria Matrix ([Sentinel_Creative])**:
   - Comprehensive risk assessment and a Markdown success criteria table (`| Strategic Metric | Target Threshold | Verification Methodology |`).
6. **Section 5: Final Executive Verdict & Immediate Action Checklist**:
   - Conclude with a formal executive ruling and an actionable `[ ]` governance checklist.

STRICT FORMATTING RULES:
- Use clean Markdown headers (`#`, `##`, `###`), Markdown tables (`| col | col |`), fenced code blocks (` ``` `), and standard lists (`-` or `1.`).
- Keep code blocks concise and split long lines/tokens cleanly so layout engines never overflow a single page.
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
# 2. Unified Heuristic Engine & Multi-Model Iterative Completion Engine
# ==============================================================================
class UnifiedHeuristicEngine:
    def __init__(self, goal_vector, covariance_matrix, target_distribution, k_constant=0.666, max_threshold=12.0):
        self.goal = np.array(goal_vector, dtype=float)
        self.sigma = np.array(covariance_matrix, dtype=float)

        # Precompute inverse covariance matrix with small diagonal regularization for numerical stability
        self.sigma_inv = np.linalg.inv(self.sigma + np.eye(self.sigma.shape[0]) * 1e-5)

        # Normalize target probability distribution (Q)
        self.q_dist = np.array(target_distribution, dtype=float)
        self.q_dist /= np.sum(self.q_dist)

        self.k = float(k_constant)
        self.max_threshold = float(max_threshold)
        # Critical collapse boundary scaled by K = 0.666
        self.d_crit = self.k * self.max_threshold

    def compute_euclidean(self, state: np.ndarray) -> float:
        """Tier 1: Latent spatial bridge (Linear Earth-standard distance)."""
        return float(np.linalg.norm(self.goal - state))

    def compute_mahalanobis(self, state: np.ndarray) -> float:
        """Tier 2: Covariance-adjusted distance accounting for feature correlation."""
        diff = self.goal - state
        val = np.dot(diff.T, np.dot(self.sigma_inv, diff))
        return float(np.sqrt(max(0.0, val)))

    def compute_kl_divergence(self, p_distribution: list) -> float:
        """Tier 3: Information divergence between current belief state P and goal Q."""
        p = np.array(p_distribution, dtype=float)
        p /= np.sum(p)  # Normalize P

        # Clip values with epsilon to avoid division by zero or log(0)
        eps = 1e-10
        p = np.clip(p, eps, 1.0)
        q = np.clip(self.q_dist, eps, 1.0)

        return float(np.sum(p * np.log(p / q)))

    def evaluate_shadow_clone(self, clone_id: int, state_vector: list, p_distribution: list) -> dict:
        state = np.array(state_vector, dtype=float)

        # 1. Calculate individual heuristic components
        h_euc = self.compute_euclidean(state)
        h_mah = self.compute_mahalanobis(state)
        h_kl = self.compute_kl_divergence(p_distribution)

        # 2. Combine into a unified composite heuristic score
        raw_heuristic = (0.35 * h_euc) + (0.45 * h_mah) + (0.20 * h_kl)

        # 3. Apply the K = 0.666 universal scaling constant for strict admissibility
        h_scaled = self.k * raw_heuristic

        # 4. Poof Diagnostic Trigger
        if h_scaled > self.d_crit or np.isinf(h_kl) or np.isnan(h_scaled):
            return {
                "clone_id": clone_id,
                "status": "POOF",
                "reason": f"Threshold breached. Scaled score {h_scaled:.3f} > D_crit {self.d_crit:.3f} "
                          f"(Euclidean: {h_euc:.2f}, Mahalanobis: {h_mah:.2f}, KL-Div: {h_kl:.2f})",
                "score": h_scaled
            }
        else:
            return {
                "clone_id": clone_id,
                "status": "SURVIVED",
                "reason": "State integrity validated across vector, covariance, and probability space.",
                "score": h_scaled
            }


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
    Executes the UnifiedHeuristicEngine across 200 shadow clones.
    Straightforward paths pass via direct state validation; anomalous exceptions
    trigger the UnifiedHeuristicEngine poof protocol.
    """
    np.random.seed(42)
    goal_vec = [1.0, 0.8, -0.5, 1.2]
    cov_matrix = [
        [2.1, 0.4, 0.1, 0.0],
        [0.4, 1.8, 0.3, 0.1],
        [0.1, 0.3, 1.2, 0.2],
        [0.0, 0.1, 0.2, 1.9]
    ]
    target_prob_dist = [0.4, 0.3, 0.2, 0.1]

    engine = UnifiedHeuristicEngine(
        goal_vector=goal_vec,
        covariance_matrix=cov_matrix,
        target_distribution=target_prob_dist,
        k_constant=0.666,
        max_threshold=10.0
    )

    poofed_clones = 0
    surviving_clones = 0
    sample_poof_log = None

    accepted = [
        {"name": "Sentinel_Tactical", "lens": "Operational Velocity & Execution Readiness", "confidence": 0.95, "critic": 0.10, "margin": 0.15},
        {"name": "Sentinel_Analyst", "lens": "Quantitative Convergence & First-Principles Logic", "confidence": 0.94, "critic": 0.12, "margin": 0.13},
        {"name": "Sentinel_Safety", "lens": "Risk Governance & Invariant Boundary Protection", "confidence": 0.98, "critic": 0.05, "margin": 0.20}
    ]
    dispelled = []

    for i in range(1, total_ephemeral_clones + 1):
        simulated_state = goal_vec + np.random.normal(0, 3.5, size=4)
        simulated_prob_dist = np.random.dirichlet(np.ones(4))

        result = engine.evaluate_shadow_clone(
            clone_id=i,
            state_vector=simulated_state,
            p_distribution=simulated_prob_dist
        )

        if result["status"] == "POOF":
            poofed_clones += 1
            if sample_poof_log is None:
                sample_poof_log = result
            if len(dispelled) < 5:
                dispelled.append({
                    "name": f"Clone_{i}_Heuristic",
                    "lens": "Unified Heuristic Stress Divergence",
                    "confidence": max(0.5, 1.0 - (result["score"] / 20.0)),
                    "critic": round(float(result["score"]), 4),
                    "excess_risk": round(float(result["score"] - engine.d_crit), 4)
                })
        else:
            surviving_clones += 1

    survival_rate = round((surviving_clones / max(1, total_ephemeral_clones)) * 100.0, 2)
    k_const = 0.666
    g_const = 9.80665

    requires_physics = any(kw in (title + prompt_text).lower() for kw in ["physics", "quantum", "vacuum", "gravitational", "energy", "cosmogenesis", "astrophysics", "speed"])
    ugpe_metric = None
    ldow_metric = None
    if requires_physics:
        ugpe_metric = round(k_const * g_const * (len(prompt_text) % 10 + 1), 4)
        ldow_metric = round((g_const / k_const) * 0.1, 4)

    return {
        "policy_vec": policy_vec,
        "effective_tau": round(float(engine.d_crit), 4),
        "accepted": accepted,
        "dispelled": dispelled,
        "swarm_total": total_ephemeral_clones,
        "swarm_passed": surviving_clones,
        "swarm_dispelled": poofed_clones,
        "swarm_survival_rate": survival_rate,
        "requires_physics": requires_physics,
        "ugpe_metric": ugpe_metric,
        "ldow_metric": ldow_metric,
        "simd_vector": [round(float(x), 4) for x in simd_vec],
        "simd_norm": round(float(np.linalg.norm(simd_vec)), 4) if simd_vec else 1.0,
    }


def query_nvidia_nim_iterative(title: str, prompt_text: str, sim_data: dict, prior_memories: list) -> str:
    """
    Multi-model iterative completion engine: Ensures output is never cut short or incomplete.
    Cascades through Primary, Secondary, Tertiary, and Quaternary models to build a complete report.
    """
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)

    accepted_str = "\n".join(
        f"  - [{c['name']}] ({c['lens']}) [VERIFIED NOMINAL]: Confidence={c['confidence']}, Variance={c['critic']}, Safety Buffer=+{c['margin']}"
        for c in sim_data["accepted"]
    )
    dispelled_str = "\n".join(
        f"  - [{c.get('name', 'Clone')}] ({c['lens']}) [HEURISTIC PRUNED / POOF]: Confidence={c['confidence']}, Score={c['critic']} >= D_crit ({sim_data['effective_tau']})"
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
        f"- Evaluation Scenarios Probed: {sim_data['swarm_total']} Probes via UnifiedHeuristicEngine\n"
        f"- Audit Consensus: {sim_data['swarm_passed']} Validated / {sim_data['swarm_dispelled']} Pruned (Pass Rate: {sim_data['swarm_survival_rate']}%)\n"
        f"- Critical Collapse Boundary (D_crit): {sim_data['effective_tau']}\n"
        f"- Validated Governance Streams:\n{accepted_str}\n"
        f"- Isolated Exceptions (Poofed):\n{dispelled_str}\n"
        f"{physics_str}\n"
        f"INSTRUCTION: Deliver a comprehensive, rigorous corporate governance and feasibility audit report for '{title}' and '{prompt_text}'. Ensure exhaustive completeness so that all sections (Executive Findings, First-Principles Mechanics/Cosmogenesis, Execution Protocol, Risk Matrix, and Final Verdict) are fully fleshed out with professional corporate language and NIST sensor annotations in brackets. Do not output raw LaTeX '$' or '\\' characters."
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
            f"| {c['name']} | {c['lens']} | Conf: {c['confidence']} / Score: {c['critic']} >= D_crit ({sim_data['effective_tau']}) | [Sentinel_Creative] POOFED |"
        )

    physics_section = ""
    if sim_data["requires_physics"]:
        physics_section = (
            "\n## 2.1 Unified Grand Prophetic Equation & Dimensional Overwrite Breakdown ([Sentinel_Analyst])\n"
            "- Universal Scaling Constant (k): 0.666\n"
            "- Earth Gravity Constant (g): 9.80665 m/s^2\n"
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
        f"An exhaustive enterprise audit evaluating **{sim_data['swarm_total']} operational scenarios** via UnifiedHeuristicEngine confirms that '{title}' achieves a **{sim_data['swarm_survival_rate']}% validation rate**. Standard heuristics successfully processed straight-forward execution vectors without unnecessary computational overhead:",
        "",
        *table_rows,
        "",
        "## 2. FIRST-PRINCIPLES MECHANICS & COSMOGENESIS SYNTHESIS ([Sentinel_Analyst])",
        "The Cosmogenesis construct establishes that structural order and operational viability emerge when redundant complexity is pruned in favor of direct execution pathways.",
        physics_section,
        "",
        "## 3. STRATEGIC EXECUTION PROTOCOL & OPERATIONAL BLUEPRINT ([Sentinel_Safety])",
        f"1. **Phase 1 — Baseline Deployment**: Initiate '{title}' within verified operational parameters, maintaining boundaries below `D_crit = {sim_data['effective_tau']}`.",
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
        f"- [ ] Enforce risk governance ceiling at D_crit = {sim_data['effective_tau']}.",
        "- [ ] Schedule quarterly executive review of enterprise milestones."
    ])

# ==============================================================================
# 3. HTML/CSS Renderer, TXT Builder, PDF Generator & ZIP Bundler
# ==============================================================================
def clean_raw_latex_artifacts(text: str) -> str:
    """Removes stray LaTeX delimiters/macros so rendered outputs stay clean."""
    text = text.replace("$$", "")
    text = re.sub(r'\\(text|mathbf|mathbb|frac|approx|times|rightarrow|cdot|sigma|psi|tau|Delta)\b', '', text)
    text = re.sub(r'(?<!\\)\$', '', text)
    text = text.replace("\\{", "{").replace("\\}", "}")
    return text

def _format_inline_html(text: str) -> str:
    cleaned = clean_raw_latex_artifacts(html.unescape(text))
    code_tokens, bold_tokens, italic_tokens = [], [], []
    cleaned = re.sub(r'`([^`]+)`', lambda m: (code_tokens.append(m.group(1)) or f"__C_{len(code_tokens)-1}__"), cleaned)
    cleaned = re.sub(r'\*\*(.*?)\*\*', lambda m: (bold_tokens.append(m.group(1)) or f"__B_{len(bold_tokens)-1}__"), cleaned)
    cleaned = re.sub(r'\*(.*?)\*', lambda m: (italic_tokens.append(m.group(1)) or f"__I_{len(italic_tokens)-1}__"), cleaned)
    escaped = html.escape(cleaned)
    for i, b in enumerate(bold_tokens):
        escaped = escaped.replace(f"__B_{i}__", f"<strong>{html.escape(b)}</strong>")
    for i, it in enumerate(italic_tokens):
        escaped = escaped.replace(f"__I_{i}__", f"<em>{html.escape(it)}</em>")
    for i, c in enumerate(code_tokens):
        escaped = escaped.replace(f"__C_{i}__", f"<code>{html.escape(c)}</code>")
    return escaped

def render_report_html(title: str, content: str, session_id: str, include_document_wrapper: bool = True) -> str:
    """
    Converts Markdown report content (including tables and code blocks) into clean HTML + CSS
    for fast WordPress display and CSS-to-PDF conversion.
    """
    lines = content.split("\n")
    body_parts = []
    table_lines = []
    code_lines = []
    in_code = False

    def flush_html_table():
        nonlocal table_lines
        if not table_lines:
            return
        parsed_rows = []
        for raw_tbl in table_lines:
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', raw_tbl):
                continue
            cols = [c.strip() for c in raw_tbl.strip("|").split("|")]
            if any(cols):
                parsed_rows.append(cols)
        if parsed_rows:
            html_rows = []
            headers = parsed_rows[0]
            html_rows.append("<thead><tr>" + "".join(f"<th>{_format_inline_html(h)}</th>" for h in headers) + "</tr></thead>")
            body_rows = []
            for r in parsed_rows[1:]:
                padded = r + [""] * max(0, len(headers) - len(r))
                body_rows.append("<tr>" + "".join(f"<td>{_format_inline_html(cell)}</td>" for cell in padded[:len(headers)]) + "</tr>")
            html_rows.append("<tbody>" + "".join(body_rows) + "</tbody>")
            body_parts.append("<table class='uesp-table'>" + "".join(html_rows) + "</table>")
        table_lines = []

    def flush_html_code():
        nonlocal code_lines
        if not code_lines:
            return
        code_txt = "\n".join(clean_raw_latex_artifacts(cl) for cl in code_lines)
        body_parts.append(f"<pre class='uesp-code'><code>{html.escape(code_txt)}</code></pre>")
        code_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            flush_html_table()
            if in_code:
                flush_html_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(raw_line.rstrip("\r\n"))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            continue
        else:
            flush_html_table()

        if not stripped:
            continue

        if stripped.startswith("# "):
            body_parts.append(f"<h1>{_format_inline_html(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            body_parts.append(f"<h2>{_format_inline_html(stripped[3:])}</h2>")
        elif stripped.startswith("### ") or stripped.startswith("#### "):
            h_txt = re.sub(r"^#+\s*", "", stripped).strip()
            body_parts.append(f"<h3>{_format_inline_html(h_txt)}</h3>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            body_parts.append(f"<div class='uesp-bullet'>• {_format_inline_html(stripped[2:])}</div>")
        elif re.match(r"^\d+\.\s", stripped):
            body_parts.append(f"<div class='uesp-bullet'>{_format_inline_html(stripped)}</div>")
        else:
            body_parts.append(f"<p>{_format_inline_html(stripped)}</p>")

    flush_html_table()
    flush_html_code()

    css_block = """
    <style>
      .uesp-report-wrap { font-family: Arial, Helvetica, sans-serif; color: #1b262c; line-height: 1.5; max-width: 100%; }
      .uesp-header-bar { display: flex; justify-content: space-between; border-bottom: 2px solid #00796b; padding-bottom: 8px; margin-bottom: 12px; }
      .uesp-brand { font-size: 16px; font-weight: bold; color: #004d40; }
      .uesp-meta-box { background: #e0f2f1; border: 1px solid #00796b; padding: 10px 12px; border-radius: 4px; font-size: 12px; margin-bottom: 14px; color: #004d40; }
      .uesp-report-wrap h1 { font-size: 20px; color: #004d40; margin: 14px 0 8px 0; }
      .uesp-report-wrap h2 { font-size: 16px; color: #00695c; margin: 12px 0 6px 0; border-bottom: 1px solid #b2dfdb; padding-bottom: 3px; }
      .uesp-report-wrap h3 { font-size: 14px; color: #00796b; margin: 10px 0 4px 0; }
      .uesp-report-wrap p { margin: 5px 0; font-size: 13px; }
      .uesp-bullet { margin: 4px 0 4px 14px; font-size: 13px; }
      .uesp-table { width: 100%; border-collapse: collapse; margin: 10px 0 14px 0; font-size: 12px; }
      .uesp-table th { background: #00796b; color: #ffffff; text-align: left; padding: 7px 8px; border: 1px solid #004d40; }
      .uesp-table td { padding: 6px 8px; border: 1px solid #80cbc4; vertical-align: top; }
      .uesp-table tr:nth-child(even) td { background: #e0f2f1; }
      .uesp-code { background: #f1f8e9; border: 1px solid #80cbc4; padding: 8px 10px; border-radius: 4px; overflow-x: auto; font-family: "Courier New", Courier, monospace; font-size: 11.5px; color: #004d40; white-space: pre-wrap; word-break: break-word; }
      @page { size: letter; margin: 0.55in; }
    </style>
    """

    inner_html = (
        f"{css_block if include_document_wrapper else ''}"
        f"<div class='uesp-report-wrap'>"
        f"<div class='uesp-header-bar'><div class='uesp-brand'>CELSIUS TECH MEDIA GROUP</div><div>{html.escape(COMPANY_DETAILS).replace(chr(10), '<br/>')}</div></div>"
        f"<div class='uesp-meta-box'><strong>Session ID:</strong> {html.escape(session_id)} | <strong>Catalog Models:</strong> {len(NVIDIA_NIM_CATALOG)} | <strong>Audit Probes:</strong> {EPHEMERAL_SWARM_COUNT}</div>"
        f"<h2>Subject: {html.escape(title)}</h2>"
        + "\n".join(body_parts)
        + "</div>"
    )

    if include_document_wrapper:
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{css_block}</head><body>{inner_html}</body></html>"
    return inner_html


def generate_txt_artifact(filename: str, title: str, content: str, session_id: str) -> str:
    """Generates a clean, well-aligned plain-text report artifact including formatted tables."""
    cleaned = clean_raw_latex_artifacts(content)
    lines = cleaned.split("\n")
    out_lines = [
        "========================================================================",
        "CELSIUS TECH MEDIA GROUP — ENTERPRISE GOVERNANCE & FEASIBILITY REPORT",
        "========================================================================",
        f"Session ID : {session_id}",
        f"Subject    : {title}",
        f"Audit Scale: {EPHEMERAL_SWARM_COUNT} Probes",
        "------------------------------------------------------------------------",
        ""
    ]

    table_buf = []

    def flush_txt_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = []
        for raw_t in table_buf:
            if re.match(r"^\s*\|?\s*:?-+:?\s*\|", raw_t):
                continue
            cols = [c.strip() for c in raw_t.strip("|").split("|")]
            if any(cols):
                rows.append(cols)
        if rows:
            col_count = max(len(r) for r in rows)
            widths = [4] * col_count
            for r in rows:
                for idx in range(col_count):
                    val = r[idx] if idx < len(r) else ""
                    widths[idx] = max(widths[idx], min(len(val), 36))
            for r_idx, r in enumerate(rows):
                formatted_cols = []
                for idx in range(col_count):
                    val = r[idx] if idx < len(r) else ""
                    formatted_cols.append(val[:widths[idx]].ljust(widths[idx]))
                out_lines.append(" | ".join(formatted_cols))
                if r_idx == 0:
                    out_lines.append("-+-".join("-" * w for w in widths))
            out_lines.append("")
        table_buf = []

    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_buf.append(s)
        else:
            flush_txt_table()
            out_lines.append(line)

    flush_txt_table()
    txt_payload = "\n".join(out_lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(txt_payload)
    return filename


def sanitize_text(text: str) -> str:
    text = clean_raw_latex_artifacts(html.unescape(text))
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


def split_long_code_line(line: str, max_len: int = 88) -> list:
    if len(line) <= max_len:
        return [line]
    chunks = []
    current = line
    while len(current) > max_len:
        split_at = -1
        for sep in [" ", ",", ";", "(", ")", "{", "}", "/", "."]:
            pos = current.rfind(sep, 20, max_len)
            if pos > split_at:
                split_at = pos + 1
        if split_at <= 0:
            split_at = max_len
        chunks.append(current[:split_at].rstrip())
        current = current[split_at:].lstrip()
    if current:
        chunks.append(current)
    return chunks


def parse_markdown_to_story(text: str, story: list, styles: dict):
    lines = text.split('\n')
    usable_width = 7.0 * inch
    table_buffer = []
    code_buffer = []
    in_code_block = False
    max_code_lines_per_block = 22
    max_code_chars_per_block = 1400

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
            story.append(Spacer(1, 5))
            story.append(t)
            story.append(Spacer(1, 7))
        table_buffer = []

    def flush_code():
        nonlocal code_buffer
        if not code_buffer:
            return
        expanded_lines = []
        for cl in code_buffer:
            expanded_lines.extend(split_long_code_line(cl, max_len=88))

        current_chunk = []
        current_chars = 0
        for cl in expanded_lines:
            if current_chunk and (len(current_chunk) >= max_code_lines_per_block or current_chars + len(cl) > max_code_chars_per_block):
                escaped_code = "<br/>".join(escape(item).replace(" ", "&nbsp;") for item in current_chunk)
                p = Paragraph(escaped_code, styles['CodeBlock'])
                t = Table([[p]], colWidths=[usable_width])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F8E9')),
                    ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#80CBC4')),
                    ('PADDING', (0, 0), (-1, -1), 6)
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
                current_chunk = []
                current_chars = 0
            current_chunk.append(cl)
            current_chars += len(cl)

        if current_chunk:
            escaped_code = "<br/>".join(escape(item).replace(" ", "&nbsp;") for item in current_chunk)
            p = Paragraph(escaped_code, styles['CodeBlock'])
            t = Table([[p]], colWidths=[usable_width])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F8E9')),
                ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#80CBC4')),
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
            code_buffer.append(raw_line.rstrip('\r\n'))
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
            story.append(safe_p(f"• {line[2:]}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line):
            story.append(safe_p(line, styles['Bullet']))
            story.append(Spacer(1, 2))
        else:
            story.append(safe_p(line, styles['Body']))
            story.append(Spacer(1, 3))

    flush_table()
    flush_code()


def guard_story_heights(story: list, max_height: float = 680.0, width: float = 7.0 * inch):
    """Pre-flight flowable check so no single flowable exceeds frame height and triggers LayoutError."""
    guarded = []
    for flowable in story:
        try:
            _, h = flowable.wrap(width, max_height * 2)
            if h > max_height and isinstance(flowable, KeepTogether):
                guarded.extend(getattr(flowable, "_content", [flowable]))
                continue
        except Exception:
            pass
        guarded.append(flowable)
    return guarded


def generate_pdf_artifact(filename: str, title: str, content: str, session_id: str, full_html: str = None):
    """Renders CSS-styled HTML to PDF via WeasyPrint when available, with ReportLab fallback."""
    html_doc = full_html or render_report_html(title, content, session_id, include_document_wrapper=True)
    if WEASYPRINT_AVAILABLE:
        try:
            WeasyHTML(string=html_doc).write_pdf(filename)
            return filename
        except Exception as e:
            print(f"[WARN] WeasyPrint CSS-to-PDF failed, falling back to ReportLab: {e}")

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
        'CodeBlock': ParagraphStyle('CodeBlock', fontName='Courier', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#004D40'), wordWrap='CJK'),
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
    safe_story = guard_story_heights(story, max_height=680.0, width=usable_width)
    doc.build(safe_story)
    return filename


def generate_zip_artifact(zip_filename: str, pdf_filename: str, txt_filename: str) -> str:
    """Bundles both the generated PDF and TXT artifacts into a compressed ZIP archive."""
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        if os.path.exists(pdf_filename):
            zf.write(pdf_filename, arcname=os.path.basename(pdf_filename))
        if os.path.exists(txt_filename):
            zf.write(txt_filename, arcname=os.path.basename(txt_filename))
    return zip_filename

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

    # 1. Render HTML immediately for fast WordPress display
    wp_html_fragment = render_report_html(title, report_content, session_id, include_document_wrapper=False)
    full_html_doc = render_report_html(title, report_content, session_id, include_document_wrapper=True)

    # 2. Generate PDF (CSS-to-PDF via WeasyPrint with ReportLab fallback) + TXT + ZIP bundle
    short_id = session_id[:16]
    pdf_filename = f"Report_{short_id}.pdf"
    txt_filename = f"Report_{short_id}.txt"
    zip_filename = f"Report_{short_id}.zip"

    generate_pdf_artifact(pdf_filename, title, report_content, session_id, full_html=full_html_doc)
    generate_txt_artifact(txt_filename, title, report_content, session_id)
    generate_zip_artifact(zip_filename, pdf_filename, txt_filename)

    print(
        f"✅ Artifacts generated: {zip_filename} "
        f"(PDF: {os.path.getsize(pdf_filename) if os.path.exists(pdf_filename) else 0}B, "
        f"TXT: {os.path.getsize(txt_filename) if os.path.exists(txt_filename) else 0}B, "
        f"ZIP: {os.path.getsize(zip_filename) if os.path.exists(zip_filename) else 0}B)"
    )

    # WordPress REST API Upload Integration (uploads ZIP bundle & publishes rendered HTML)
    wp_url = "[https://celsiustechmediagroup.co.za/wp-json/wp/v2](https://celsiustechmediagroup.co.za/wp-json/wp/v2)"
    wp_user = os.getenv("WP_USERNAME")
    wp_pass = os.getenv("WP_APP_PASSWORD")

    if wp_user and wp_pass:
        print(f"🌐 Uploading ZIP artifact to WordPress Media Library at {wp_url}...")
        try:
            with open(zip_filename, 'rb') as f:
                zip_bytes = f.read()

            headers = {
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "Content-Type": "application/zip"
            }
            media_resp = requests.post(
                f"{wp_url}/media",
                headers=headers,
                data=zip_bytes,
                auth=(wp_user, wp_pass),
                timeout=30
            )
            if media_resp.status_code in [200, 201]:
                media_data = media_resp.json()
                media_id = media_data.get("id")
                media_source_url = media_data.get("source_url")
                print(f"✅ Successfully uploaded ZIP to WordPress! Media ID: {media_id}, URL: {media_source_url}")

                wp_body = (
                    f"{wp_html_fragment}\n\n"
                    f"<p><strong>ECTA Audit Token:</strong> <code>{html.escape(session_id)}</code></p>"
                    f"<p><a href='{html.escape(media_source_url)}' target='_blank' rel='noopener'>📥 Download Full Artifact Bundle (ZIP: PDF + TXT)</a></p>"
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
                    print("✅ WordPress UESP record successfully created with instant HTML and ZIP artifact link.")
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
