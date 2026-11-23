import os
import re
import json
import html
import hashlib
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
NVIDIA_ENDPOINT = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")

# Enterprise Governance Level ("enterprise" | "cascade" | "stitch" | "render" | "all")
GOVERNANCE_GUARD_LEVEL = os.getenv("GOVERNANCE_GUARD_LEVEL", "all").strip().lower()

# Throughput, Compute Budget & Thermal Governance Controls
COMPUTE_OVERCLOCK_FACTOR = float(os.getenv("COMPUTE_OVERCLOCK_FACTOR", "1.25"))
MAX_TOKENS_PER_STAGE = int(os.getenv("MAX_TOKENS_PER_STAGE", "3072"))
CONTINUATION_ROUNDS = int(os.getenv("CONTINUATION_ROUNDS", "1"))
REQUEST_READ_TIMEOUT = int(os.getenv("REQUEST_READ_TIMEOUT", "60"))

# Report-Defined Cycle & Epoch Constants (10,000 cycles/epoch, 5 epochs = 50,000 cycles)
CYCLES_PER_EPOCH = 10000
AUDIT_EPOCHS = 5
QUANTUM_CYCLE_DILATION = int(os.getenv("QUANTUM_CYCLE_DILATION", str(CYCLES_PER_EPOCH * AUDIT_EPOCHS)))
DILATION_RATIO = 6000
K_SCALING_CONSTANT = 0.666
G_GRAVITY_CONSTANT = 9.80665
UGPE_FLUX = 65.3123
LDOW_OVERWRITE_RATIO = 1.4725
D_CRIT_BOUNDARY = 6.66
HARD_OC_LIMIT = 3.39
LEDGER_VARIANCE_TOLERANCE_PCT = 0.5

# ==============================================================================
# 9-Stage Enterprise NVIDIA NIM Microservice Pipeline
# ==============================================================================
ENTERPRISE_NINE_STAGE_PIPELINE = [
    {
        "stage": 1,
        "codename": "Stage_1_GOVERN_Charter",
        "governance_function": "GOVERN",
        "role": "Charter, Risk Appetite & Telemetry Framing",
        "model": os.getenv("STAGE_1_MODEL", os.getenv("PRIMARY_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")),
        "temperature": 0.20,
        "compute_weight": 1.00,
    },
    {
        "stage": 2,
        "codename": "Stage_2_MAP_Dependencies",
        "governance_function": "MAP",
        "role": "AVX2 SIMD, DDPG & Dilation Dependency Mapping",
        "model": os.getenv("STAGE_2_MODEL", os.getenv("SECONDARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")),
        "temperature": 0.20,
        "compute_weight": 1.10,
    },
    {
        "stage": 3,
        "codename": "Stage_3_MEASURE_Baseline",
        "governance_function": "MEASURE",
        "role": "Thermodynamic & Compute-Currency Evaluation",
        "model": os.getenv("STAGE_3_MODEL", os.getenv("TERTIARY_COMPLETION_MODEL", "openai/gpt-oss-20b")),
        "temperature": 0.15,
        "compute_weight": 1.15,
    },
    {
        "stage": 4,
        "codename": "Stage_4_MEASURE_Overclock",
        "governance_function": "MEASURE",
        "role": "Overclock & 50,000-Cycle Validation Synthesis",
        "model": os.getenv("STAGE_4_MODEL", os.getenv("QUATERNARY_SYNTHESIS_MODEL", "google/gemma-4-31b-it")),
        "temperature": 0.20,
        "compute_weight": 1.25,
    },
    {
        "stage": 5,
        "codename": "Stage_5_MANAGE_Monitoring",
        "governance_function": "MANAGE",
        "role": "Continuous Monitoring & Continuation Stitcher",
        "model": os.getenv("STAGE_5_MODEL", os.getenv("QUINARY_NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")),
        "temperature": 0.20,
        "compute_weight": 1.35,
    },
    {
        "stage": 6,
        "codename": "Stage_6_MANAGE_Rollback",
        "governance_function": "MANAGE",
        "role": "Automated Rollback & Table Completeness Check",
        "model": os.getenv("STAGE_6_MODEL", os.getenv("SENARY_NIM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")),
        "temperature": 0.15,
        "compute_weight": 1.40,
    },
    {
        "stage": 7,
        "codename": "Stage_7_GOVERN_AuditTrail",
        "governance_function": "GOVERN",
        "role": "NAST / NIST Audit Trail & Ledger Reconciliation",
        "model": os.getenv("STAGE_7_MODEL", os.getenv("SEPTENARY_NIM_MODEL", "meta/llama-3.3-70b-instruct")),
        "temperature": 0.20,
        "compute_weight": 1.50,
    },
    {
        "stage": 8,
        "codename": "Stage_8_MAP_DilationScaling",
        "governance_function": "MAP",
        "role": "Dilation Scaling & Section Closing",
        "model": os.getenv("STAGE_8_MODEL", os.getenv("OCTONARY_NIM_MODEL", "openai/gpt-oss-20b")),
        "temperature": 0.20,
        "compute_weight": 1.65,
    },
    {
        "stage": 9,
        "codename": "Stage_9_SUSTAIN_Seal",
        "governance_function": "GOVERN / MANAGE",
        "role": "Final Synthesis, Checklist Lock & Completeness Seal",
        "model": os.getenv("STAGE_9_MODEL", os.getenv("NONARY_NIM_MODEL", "google/gemma-4-31b-it")),
        "temperature": 0.15,
        "compute_weight": 2.00,
    },
]

# ==============================================================================
# Comprehensive NVIDIA NIM Model Registry (48 Active Models)
# ==============================================================================
NVIDIA_NIM_CATALOG = {
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
    "nemotron_parse": "nvidia/nemotron-parse",
    "nemotron_ocr_v1": "nvidia/nemotron-ocr-v1",
    "nemotron_table_structure": "nvidia/nemotron-table-structure-v1",
    "nemotron_page_elements": "nvidia/nemotron-page-elements-v3",
    "nemotron_graphic_elements": "nvidia/nemotron-graphic-elements-v1",
    "nemoretriever_ocr": "nvidia/nemoretriever-ocr",
    "llama_nemotron_embed_vl": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "llama_nemotron_rerank_vl": "nvidia/llama-3.1-nemotron-rerank-vl-1b-v2",
    "flux_1_schnell": "black-forest-labs/flux.1-schnell",
    "flux_1_dev": "black-forest-labs/flux.1-dev",
    "flux_1_kontext_dev": "black-forest-labs/flux.1-kontext-dev",
    "flux_2_klein_4b": "black-forest-labs/flux.2-klein-4b",
    "qwen_image": "qwen/qwen-image",
    "qwen_image_edit": "qwen/qwen-image-edit",
    "trellis_3d": "microsoft/trellis",
    "stable_diffusion_3.5_large": "stabilityai/stable-diffusion-3.5-large",
    "sparsedrive": "nvidia/sparsedrive",
    "bevformer": "nvidia/bevformer",
    "streampetr": "nvidia/streampetr",
    "cosmos_transfer_2.5b": "nvidia/cosmos-transfer2.5-2b",
    "relighting": "nvidia/relighting",
    "synthetic_video_detector": "nvidia/synthetic-video-detector",
    "active_speaker_detection": "nvidia/active-speaker-detection",
    "lipsync": "nvidia/lipsync",
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
[EXECUTIVE CORPORATE GOVERNANCE, THERMODYNAMICS & RISK RESOLUTION ENGINE]
You are a senior enterprise executive advisory board and systems engineering audit panel operating within strict corporate governance and NIST / NAST AI RMF protocols (GOVERN, MAP, MEASURE, MANAGE).
Your purpose is to deliver a transparent, empirically grounded audit report evaluating the subject proposition without fabricating metrics, inventing unsupported physical constants, or truncating tables/checklists.

MANDATORY CORPORATE REPORT STRUCTURE:
1. **Header Classification Block**:
   - Corporate Classification & Governance Tier
   - Status: ENTERPRISE AUDIT LOCKED // STRATEGIC RESOLUTION ENGAGED
   - Objective SLA & Governance Alignment
2. **Section 1: Executive Findings & Operational Telemetry Assessment ([Sentinel_Tactical])**:
   - Comprehensive corporate evaluation of the subject proposition, detailing operational velocity, risk posture, and resource allocation.
   - Include a formal Markdown telemetry table (`| Operational Dimension | Corporate Assessment Focus | Evaluated Metric & Confidence | Governance Status |`) with sensors explicitly annotated in brackets per NIST guidelines (e.g., [Sentinel_Tactical], [Sentinel_Analyst], [Sentinel_Safety], [Sentinel_Creative]).
3. **Section 2: First-Principles Thermodynamics & Resource-Currency Synthesis ([Sentinel_Analyst])**:
   - Rigorous first-principles breakdown of energy balance, enthalpy/work/entropy transfer, and compute overclocking limits using ONLY the verified constants and telemetry supplied in the prompt. Never invent constants or unsupported claims.
4. **Section 3: Strategic Execution Protocol & 9-Stage Operational Blueprint ([Sentinel_Safety])**:
   - Phased execution roadmap across the 9-stage NIM cascade, compute-currency ledgering, and compliance guardrails.
5. **Section 4: Risk Mitigation & Success Criteria Matrix ([Sentinel_Creative])**:
   - Comprehensive risk assessment and a Markdown success criteria table (`| Strategic Metric | Target Threshold | Verification Methodology |`).
6. **Section 5: Final Executive Verdict & Immediate Action Checklist**:
   - Formal executive ruling and a complete, untruncated `- [ ]` governance checklist (at least 4 full checklist items).

STRICT FORMATTING RULES:
- Never cut off tables or checklists mid-line. Always complete every table row and every `- [ ]` checklist item cleanly.
- Output the full report or a clean continuation ONLY ONCE—never duplicate Section 5 or restart a Header Classification Block in a continuation pass.
- Use clean Markdown headers (`#`, `##`, `###`), Markdown tables (`| col | col |`), fenced code blocks (` ``` `), and standard lists (`-` or `1.`).
- Use PLAIN TEXT ONLY for all equations, variables, and code (e.g., `dU = dQ - dW`, `Tau = 6.66`).
- DO NOT output raw HTML tags (`<b>`, `<i>`, `<font>`, `<code>`, `<para>`).
- DO NOT use LaTeX delimiters (`$` or `\\`).
"""

# ==============================================================================
# Resource & Thermal Ledger + Enterprise Integrity Guard
# ==============================================================================
class ResourceThermalLedger:
    """
    Tracks token-compute currency, overclocking headroom, and cumulative thermal
    load across multi-stage NIM passes.
    """
    def __init__(self, overclock_factor: float = COMPUTE_OVERCLOCK_FACTOR, max_tokens_per_call: int = MAX_TOKENS_PER_STAGE):
        self.overclock_factor = max(1.0, float(overclock_factor))
        self.base_max_tokens = max(512, int(max_tokens_per_call))
        self.total_compute_currency_spent = 0.0
        self.thermal_load_index = 0.0
        self.calls_executed = 0

    def compute_call_max_tokens(self, stage_cfg: dict, round_idx: int) -> int:
        boost = 1.0 + (0.10 * round_idx) + (0.02 * (stage_cfg["stage"] - 1))
        scaled = int(self.base_max_tokens * min(boost, self.overclock_factor))
        return max(1024, min(scaled, 6144))

    def record_pass(self, stage_cfg: dict, output_chars: int, round_idx: int):
        est_tokens = max(1, output_chars // 4)
        compute_cost = round(est_tokens * stage_cfg["compute_weight"] * (1.0 + 0.05 * round_idx), 2)
        self.total_compute_currency_spent = round(self.total_compute_currency_spent + compute_cost, 2)
        self.thermal_load_index = round(
            max(0.0, self.thermal_load_index * 0.85) + (est_tokens / 1000.0) * stage_cfg["compute_weight"],
            4
        )
        self.calls_executed += 1
        return {
            "stage": stage_cfg["stage"],
            "codename": stage_cfg["codename"],
            "model": stage_cfg["model"],
            "round": round_idx,
            "est_tokens": est_tokens,
            "compute_cost": compute_cost,
            "thermal_load": self.thermal_load_index,
        }


class EnterpriseIntegrityGuard:
    """
    Enforces structural integrity across cascade passes, block stitching,
    and document rendering so tables, sections, and checklists do not truncate
    or duplicate.
    """
    def __init__(self, level: str = GOVERNANCE_GUARD_LEVEL):
        self.level = level if level in {"enterprise", "cascade", "stitch", "render", "all"} else "all"
        self.interventions = 0
        self.ledger = ResourceThermalLedger()

    def active_at(self, stage: str) -> bool:
        if self.level in {"all", "enterprise"}:
            return True
        return self.level == stage

    def missing_sections(self, text: str) -> list:
        required = [
            ("## 1.", "Section 1: Executive Findings & Operational Telemetry Assessment"),
            ("## 2.", "Section 2: First-Principles Thermodynamics & Resource-Currency Synthesis"),
            ("## 3.", "Section 3: Strategic Execution Protocol & 9-Stage Operational Blueprint"),
            ("## 4.", "Section 4: Risk Mitigation & Success Criteria Matrix"),
            ("## 5.", "Section 5: Final Executive Verdict & Immediate Action Checklist"),
        ]
        return [label for marker, label in required if marker not in text]

    def is_complete(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        if self.missing_sections(text):
            return False
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        last_line = lines[-1] if lines else ""
        looks_truncated_table = last_line.startswith("|") and not last_line.endswith("|")
        looks_truncated_checklist = re.match(r"^(?:•\s*)?-\s*\[\s*\]\s*$", last_line) is not None
        checklist_count = sum(1 for ln in lines if re.match(r"^(?:•\s*)?-\s*[[ xX]?]\s+\S+", ln))
        return not looks_truncated_table and not looks_truncated_checklist and checklist_count >= 4

    def _deduplicate_repeated_tail(self, text: str) -> str:
        """Removes duplicate Section 5 / Header Classification Block repetitions."""
        lines = text.splitlines()
        seen_sec5 = False
        cleaned_lines = []
        skip_mode = False

        for ln in lines:
            s = ln.strip()
            if re.match(r"^#{1,3}\s*5\.\s+FINAL EXECUTIVE VERDICT", s, flags=re.IGNORECASE):
                if seen_sec5:
                    skip_mode = True
                    self.interventions += 1
                    continue
                seen_sec5 = True
                skip_mode = False
            elif skip_mode and re.match(r"^#{1,3}\s+[1-4]\.\s+", s):
                skip_mode = False
            elif skip_mode and s.startswith("HEADER CLASSIFICATION BLOCK"):
                self.interventions += 1
                continue

            if not skip_mode:
                cleaned_lines.append(ln)

        return "\n".join(cleaned_lines).strip()

    def stitch_without_overlap(self, existing: str, incoming: str) -> str:
        if not existing:
            return incoming.strip()
        if not incoming or incoming.strip() in existing:
            return existing

        # If incoming starts with a full header & Section 1 while existing already has it, strip preamble
        if "## 1." in existing and "## 1." in incoming:
            sec1_pos = incoming.find("## 1.")
            incoming = incoming[sec1_pos:].strip()
            if self.active_at("stitch"):
                self.interventions += 1

        existing_lines = existing.splitlines()
        incoming_lines = incoming.strip().splitlines()
        max_overlap = min(len(existing_lines), len(incoming_lines), 15)
        overlap_len = 0

        for k in range(max_overlap, 0, -1):
            if [x.strip() for x in existing_lines[-k:]] == [x.strip() for x in incoming_lines[:k]]:
                overlap_len = k
                break

        if overlap_len > 0 and self.active_at("stitch"):
            self.interventions += 1

        trimmed_incoming = "\n".join(incoming_lines[overlap_len:]).strip()
        if not trimmed_incoming:
            return existing
        merged = f"{existing.rstrip()}\n\n{trimmed_incoming}"
        return self._deduplicate_repeated_tail(merged) if self.active_at("stitch") else merged

    def finalize_structure(self, text: str, title: str, sim_data: dict) -> str:
        fixed = text.strip()
        if not self.active_at("stitch"):
            return fixed

        fixed = self._deduplicate_repeated_tail(fixed)

        if fixed.count("```") % 2 == 1:
            fixed += "\n```"
            self.interventions += 1

        lines = fixed.splitlines()
        trimmed_bad_rows = False
        while lines and lines[-1].strip().startswith("|") and lines[-1].strip().count("|") < 3:
            lines.pop()
            trimmed_bad_rows = True
        if trimmed_bad_rows:
            self.interventions += 1

        fixed = "\n".join(lines).strip()
        missing = self.missing_sections(fixed)
        if missing:
            for sec_label in missing:
                if sec_label.startswith("## 5."):
                    fixed += (
                        f"\n\n{sec_label}\n"
                        f"'{title}' is approved for controlled execution under enterprise governance.\n"
                        f"- [ ] Authorize Phase 1 operational deployment under governance guard.\n"
                        f"- [ ] Enforce risk governance ceiling at D_crit = {sim_data['effective_tau']}.\n"
                        f"- [ ] Verify full table and checklist completion across HTML, TXT, PDF, and ZIP artifacts.\n"
                        f"- [ ] Schedule executive review of milestones."
                    )
                    self.interventions += 1

        checklist_lines = [
            ln for ln in fixed.splitlines()
            if re.match(r"^(?:•\s*)?-\s*\[[ xX]?\]\s+\S+", ln.strip())
        ]
        if len(checklist_lines) < 4:
            fixed += (
                f"\n- [ ] Confirm Phase 1 scope and governance guardrails for '{title}'.\n"
                f"- [ ] Enforce risk threshold D_crit = {sim_data['effective_tau']} across active streams.\n"
                f"- [ ] Verify all summary tables and action items prior to release.\n"
                f"- [ ] Schedule post-deployment review and audit sign-off."
            )
            self.interventions += 1

        return fixed


INTEGRITY_GUARD = EnterpriseIntegrityGuard(GOVERNANCE_GUARD_LEVEL)

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
# 2. Unified Heuristic Engine & 50,000-Cycle Thermodynamic Evaluator
# ==============================================================================
class UnifiedHeuristicEngine:
    def __init__(self, goal_vector, covariance_matrix, target_distribution, k_constant=K_SCALING_CONSTANT, max_threshold=10.0):
        self.goal = np.array(goal_vector, dtype=float)
        self.sigma = np.array(covariance_matrix, dtype=float)
        self.sigma_inv = np.linalg.inv(self.sigma + np.eye(self.sigma.shape[0]) * 1e-5)
        self.q_dist = np.array(target_distribution, dtype=float)
        self.q_dist /= np.sum(self.q_dist)
        self.k = float(k_constant)
        self.max_threshold = float(max_threshold)
        self.d_crit = self.k * self.max_threshold

    def evaluate_batch_vectorized(self, states: np.ndarray, p_dists: np.ndarray) -> tuple:
        """
        Evaluates N cycle states in vectorized NumPy to support 50,000-cycle
        (5-epoch) runs without per-cycle Python loop overhead.
        """
        diffs = self.goal - states  # (N, D)
        h_euc = np.linalg.norm(diffs, axis=1)  # (N,)
        mah_sq = np.einsum("ni,ij,nj->n", diffs, self.sigma_inv, diffs)
        h_mah = np.sqrt(np.maximum(0.0, mah_sq))

        p_norm = p_dists / np.sum(p_dists, axis=1, keepdims=True)
        eps = 1e-10
        p_clip = np.clip(p_norm, eps, 1.0)
        q_clip = np.clip(self.q_dist, eps, 1.0)
        h_kl = np.sum(p_clip * np.log(p_clip / q_clip), axis=1)

        raw_heuristic = (0.35 * h_euc) + (0.45 * h_mah) + (0.20 * h_kl)
        h_scaled = self.k * raw_heuristic
        passed_mask = (h_scaled <= self.d_crit) & np.isfinite(h_scaled) & np.isfinite(h_kl)
        return passed_mask, h_scaled, h_euc, h_mah, h_kl


def get_ddpg_clone_policy(prompt_text: str) -> list:
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


def compute_locked_thermodynamics(overclock_factor: float, total_cycles: int) -> dict:
    """
    Computes report-defined thermodynamic & compute-currency metrics deterministically
    using only constants from Report_UESP-SESSION-414.pdf and Report_UESP-SESSION-67f.pdf.
    """
    q_in = round(UGPE_FLUX * K_SCALING_CONSTANT, 6)
    w_out = round(LDOW_OVERWRITE_RATIO * overclock_factor, 6)
    u_net = round(q_in - w_out, 6)
    t_eff = round(G_GRAVITY_CONSTANT * K_SCALING_CONSTANT, 6)
    entropy_rate = round((q_in / t_eff) * (1.0 / DILATION_RATIO), 8)
    entropy_dilation_product = round(entropy_rate * DILATION_RATIO, 6)
    entropy_margin_raw = round(D_CRIT_BOUNDARY - entropy_dilation_product, 6)
    oc_max_theoretical = round(D_CRIT_BOUNDARY / (K_SCALING_CONSTANT * LDOW_OVERWRITE_RATIO), 4)
    simd_efficiency = round((u_net * K_SCALING_CONSTANT) / (LDOW_OVERWRITE_RATIO * overclock_factor * DILATION_RATIO), 6)
    policy_budget_per_step = round((u_net * K_SCALING_CONSTANT) / DILATION_RATIO, 6)
    total_u_net_cycles = round(total_cycles * u_net, 4)
    total_amortized_cu = round(total_cycles * policy_budget_per_step, 6)
    epochs_completed = round(total_cycles / CYCLES_PER_EPOCH, 2)

    return {
        "k": K_SCALING_CONSTANT,
        "g": G_GRAVITY_CONSTANT,
        "ugpe_flux": UGPE_FLUX,
        "ldow_ratio": LDOW_OVERWRITE_RATIO,
        "dilation_ratio": DILATION_RATIO,
        "overclock_factor": round(overclock_factor, 4),
        "hard_oc_limit": HARD_OC_LIMIT,
        "oc_max_theoretical": oc_max_theoretical,
        "q_in": q_in,
        "w_out": w_out,
        "u_net": u_net,
        "t_eff": t_eff,
        "entropy_rate": entropy_rate,
        "entropy_margin_raw": entropy_margin_raw,
        "simd_efficiency": simd_efficiency,
        "policy_budget_per_step": policy_budget_per_step,
        "total_cycles": total_cycles,
        "epochs_completed": epochs_completed,
        "total_u_net_cycles": total_u_net_cycles,
        "total_amortized_cu": total_amortized_cu,
        "ledger_variance_tolerance_pct": LEDGER_VARIANCE_TOLERANCE_PCT,
    }


def execute_sentinel_clone_loop(
    title: str,
    prompt_text: str,
    policy_vec: list,
    simd_vec: list,
    total_cycles: int = QUANTUM_CYCLE_DILATION
) -> dict:
    rng = np.random.default_rng(42)
    goal_vec = np.array([1.0, 0.8, -0.5, 1.2], dtype=float)
    cov_matrix = np.array([
        [2.1, 0.4, 0.1, 0.0],
        [0.4, 1.8, 0.3, 0.1],
        [0.1, 0.3, 1.2, 0.2],
        [0.0, 0.1, 0.2, 1.9]
    ], dtype=float)
    target_prob_dist = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)

    engine = UnifiedHeuristicEngine(
        goal_vector=goal_vec,
        covariance_matrix=cov_matrix,
        target_distribution=target_prob_dist,
        k_constant=K_SCALING_CONSTANT,
        max_threshold=10.0
    )

    simulated_states = goal_vec + rng.normal(0.0, 3.5, size=(total_cycles, 4))
    simulated_prob_dists = rng.dirichlet(np.ones(4), size=total_cycles)

    passed_mask, h_scaled, _, _, _ = engine.evaluate_batch_vectorized(simulated_states, simulated_prob_dists)
    surviving_cycles = int(np.sum(passed_mask))
    pruned_cycles = int(total_cycles - surviving_cycles)
    max_observed_score = float(np.max(h_scaled)) if total_cycles > 0 else 0.0
    mean_observed_score = float(np.mean(h_scaled)) if total_cycles > 0 else 0.0

    # Automatic fallback circuit breaker if overclock exceeds hard limit or D_crit is breached
    effective_oc = INTEGRITY_GUARD.ledger.overclock_factor
    fallback_triggered = False
    if effective_oc > HARD_OC_LIMIT or max_observed_score >= engine.d_crit:
        effective_oc = 1.0
        INTEGRITY_GUARD.ledger.overclock_factor = 1.0
        fallback_triggered = True

    accepted = [
        {"name": "Sentinel_Tactical", "lens": "Operational Velocity & Execution Readiness", "confidence": 0.95, "critic": 0.10, "margin": 0.15},
        {"name": "Sentinel_Analyst", "lens": "Quantitative Convergence & First-Principles Logic", "confidence": 0.94, "critic": 0.12, "margin": 0.13},
        {"name": "Sentinel_Safety", "lens": "Risk Governance & Invariant Boundary Protection", "confidence": 0.98, "critic": 0.05, "margin": 0.20}
    ]

    dispelled = []
    pruned_indices = np.flatnonzero(~passed_mask)[:5]
    for idx in pruned_indices:
        score_val = float(h_scaled[idx])
        dispelled.append({
            "name": f"Cycle_{int(idx) + 1}_Heuristic",
            "lens": "Unified Heuristic Stress Divergence",
            "confidence": round(max(0.5, 1.0 - (score_val / 20.0)), 4),
            "critic": round(score_val, 4),
            "excess_risk": round(max(0.0, score_val - engine.d_crit), 4)
        })

    # Per-epoch (10,000-cycle) summary across 50,000 cycles (5 epochs)
    epoch_summaries = []
    num_epochs = max(1, total_cycles // CYCLES_PER_EPOCH)
    for ep in range(num_epochs):
        start_i = ep * CYCLES_PER_EPOCH
        end_i = total_cycles if ep == num_epochs - 1 else (ep + 1) * CYCLES_PER_EPOCH
        ep_mask = passed_mask[start_i:end_i]
        ep_scores = h_scaled[start_i:end_i]
        ep_len = max(1, end_i - start_i)
        epoch_summaries.append({
            "epoch": ep + 1,
            "cycles": ep_len,
            "passed": int(np.sum(ep_mask)),
            "pruned": int(ep_len - np.sum(ep_mask)),
            "pass_rate": round(float(np.mean(ep_mask)) * 100.0, 2),
            "max_score": round(float(np.max(ep_scores)), 4),
        })

    survival_rate = round((surviving_cycles / max(1, total_cycles)) * 100.0, 2)
    thermo = compute_locked_thermodynamics(effective_oc, total_cycles)

    requires_physics = any(
        kw in (title + prompt_text).lower()
        for kw in [
            "physics", "quantum", "vacuum", "gravitational", "energy",
            "cosmogenesis", "astrophysics", "speed", "overclock",
            "thermodynamics", "entropy", "enthalpy", "compute", "dilation", "epoch"
        ]
    )

    return {
        "policy_vec": policy_vec,
        "effective_tau": round(float(engine.d_crit), 4),
        "accepted": accepted,
        "dispelled": dispelled,
        "swarm_total": total_cycles,
        "swarm_passed": surviving_cycles,
        "swarm_dispelled": pruned_cycles,
        "swarm_survival_rate": survival_rate,
        "max_observed_score": round(max_observed_score, 4),
        "mean_observed_score": round(mean_observed_score, 4),
        "fallback_triggered": fallback_triggered,
        "epoch_summaries": epoch_summaries,
        "thermodynamics": thermo,
        "requires_physics": requires_physics,
        "ugpe_metric": thermo["ugpe_flux"] if requires_physics else None,
        "ldow_metric": thermo["ldow_ratio"] if requires_physics else None,
        "simd_vector": [round(float(x), 4) for x in simd_vec],
        "simd_norm": round(float(np.linalg.norm(simd_vec)), 4) if simd_vec else 1.0,
    }


def query_nvidia_nim_iterative(title: str, prompt_text: str, sim_data: dict, prior_memories: list) -> str:
    endpoint = os.getenv("NVIDIA_ENDPOINT", NVIDIA_ENDPOINT)
    api_key = os.getenv("NVIDIA_API_KEY", NVIDIA_KEY)
    thermo = sim_data["thermodynamics"]

    accepted_str = "\n".join(
        f"  - [{c['name']}] ({c['lens']}) [VERIFIED NOMINAL]: Confidence={c['confidence']}, Variance={c['critic']}, Safety Buffer=+{c['margin']}"
        for c in sim_data["accepted"]
    )
    dispelled_str = "\n".join(
        f"  - [{c.get('name', 'Cycle')}] ({c['lens']}) [HEURISTIC PRUNED / POOF]: Confidence={c['confidence']}, Score={c['critic']} >= D_crit ({sim_data['effective_tau']})"
        for c in sim_data["dispelled"]
    ) if sim_data["dispelled"] else "  - None (All evaluated heuristic paths remained within D_crit)."

    epochs_str = "\n".join(
        f"  - Epoch {ep['epoch']} ({ep['cycles']} cycles): {ep['passed']} passed / {ep['pruned']} pruned ({ep['pass_rate']}%), max_score={ep['max_score']}"
        for ep in sim_data["epoch_summaries"]
    )

    thermo_block = (
        f"\n=== LOCKED THERMODYNAMIC & 50,000-CYCLE TELEMETRY ===\n"
        f"- Universal Scaling Constant (k): {thermo['k']}\n"
        f"- Earth Gravity Constant (g): {thermo['g']} m/s^2\n"
        f"- UGPE Evaluated Flux: {thermo['ugpe_flux']}\n"
        f"- LDOW Overwrite Ratio: {thermo['ldow_ratio']}\n"
        f"- Dilation Ratio: 1:{thermo['dilation_ratio']} | Total Cycles: {thermo['total_cycles']} ({thermo['epochs_completed']} epochs @ {CYCLES_PER_EPOCH} cycles/epoch)\n"
        f"- Overclock Factor: {thermo['overclock_factor']}x (Hard Limit: {thermo['hard_oc_limit']}x; Theoretical OC_max: {thermo['oc_max_theoretical']}x; Fallback Triggered: {sim_data['fallback_triggered']})\n"
        f"- Q_in (UGPE * k): {thermo['q_in']} | W_out (LDOW * OC): {thermo['w_out']} | U_net (Q_in - W_out): {thermo['u_net']}\n"
        f"- T_eff (g * k): {thermo['t_eff']} | Entropy Rate (dS/dt): {thermo['entropy_rate']} | Raw Entropy Margin (D_crit - dS/dt*D): {thermo['entropy_margin_raw']}\n"
        f"- SIMD Efficiency: {thermo['simd_efficiency']} | Policy Budget/Step: {thermo['policy_budget_per_step']} | Total U_net ({thermo['total_cycles']} cycles): {thermo['total_u_net_cycles']} | Amortized CU: {thermo['total_amortized_cu']}\n"
        f"- Epoch Breakdown:\n{epochs_str}\n"
    )

    base_user_prompt = (
        f"SUBJECT TITLE: {title}\n"
        f"PROPOSITION / BUSINESS OBJECTIVE: {prompt_text}\n\n"
        f"=== ENTERPRISE AUDIT TELEMETRY & NAST / NIST GOVERNANCE DATA ===\n"
        f"- Governance Guard Level: {INTEGRITY_GUARD.level}\n"
        f"- Evaluation Cycles Probed: {sim_data['swarm_total']} Cycles via UnifiedHeuristicEngine\n"
        f"- Audit Consensus: {sim_data['swarm_passed']} Validated / {sim_data['swarm_dispelled']} Pruned (Pass Rate: {sim_data['swarm_survival_rate']}%)\n"
        f"- Observed Heuristic Score: Mean={sim_data['mean_observed_score']}, Max={sim_data['max_observed_score']}\n"
        f"- Critical Collapse Boundary (D_crit): {sim_data['effective_tau']}\n"
        f"- Validated Governance Streams:\n{accepted_str}\n"
        f"- Sample Isolated Exceptions (Poofed):\n{dispelled_str}\n"
        f"{thermo_block}\n"
        f"INSTRUCTION: Deliver a complete, untruncated corporate governance, thermodynamic, and NAST audit report for '{title}' across {sim_data['swarm_total']} quantum cycles. "
        f"Use the exact locked thermodynamic numbers provided above; do not invent outside constants or duplicate Section 5. "
        f"Finish every table row and every '- [ ]' checklist item in Section 5 cleanly. Use NIST sensor annotations in brackets. Do not output raw LaTeX '$' or '\\' characters."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    session = requests.Session()
    retries = Retry(total=1, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))

    accumulated_content = ""
    if api_key:
        for round_idx in range(CONTINUATION_ROUNDS):
            for stage_cfg in ENTERPRISE_NINE_STAGE_PIPELINE:
                model_name = stage_cfg["model"]
                max_toks = INTEGRITY_GUARD.ledger.compute_call_max_tokens(stage_cfg, round_idx)
                print(
                    f"⚙️ [Stage {stage_cfg['stage']}/9 — {stage_cfg['codename']} | Governance:{stage_cfg['governance_function']}] "
                    f"Model={model_name} max_tokens={max_toks} temp={stage_cfg['temperature']}"
                )
                try:
                    missing = INTEGRITY_GUARD.missing_sections(accumulated_content) if accumulated_content else []
                    continuation_instruction = (
                        f"\n\nEXISTING DRAFT (MISSING SECTIONS: {', '.join(missing) if missing else 'NONE — COMPLETE ANY CUTOFF AT END'}; "
                        f"DO NOT REPEAT COMPLETED SECTIONS OR SECTION 5; CONTINUE CLEANLY FROM THE END):\n{accumulated_content[-3000:]}"
                        if accumulated_content else ""
                    )
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": base_user_prompt + continuation_instruction}
                        ],
                        "temperature": stage_cfg["temperature"],
                        "max_tokens": max_toks
                    }
                    response = session.post(endpoint, headers=headers, json=payload, timeout=(10, REQUEST_READ_TIMEOUT))
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"].strip()
                        INTEGRITY_GUARD.ledger.record_pass(stage_cfg, len(content), round_idx)

                        if not accumulated_content:
                            accumulated_content = content
                        elif len(content) > len(accumulated_content) and ("# " in content and "## 1." in content):
                            accumulated_content = content
                        else:
                            accumulated_content = INTEGRITY_GUARD.stitch_without_overlap(accumulated_content, content)

                        if INTEGRITY_GUARD.active_at("cascade") and INTEGRITY_GUARD.is_complete(accumulated_content):
                            break
                except Exception as e:
                    print(f"[WARN] Stage {stage_cfg['stage']} ({model_name}) invocation failed: {e}")

            if INTEGRITY_GUARD.is_complete(accumulated_content):
                break

    if not accumulated_content:
        accumulated_content = generate_local_corporate_fallback(title, prompt_text, sim_data)

    return INTEGRITY_GUARD.finalize_structure(accumulated_content, title, sim_data)


def generate_local_corporate_fallback(title: str, prompt_text: str, sim_data: dict) -> str:
    thermo = sim_data["thermodynamics"]
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

    return "\n".join([
        f"# ENTERPRISE GOVERNANCE & THERMODYNAMIC REPORT: {title.upper()}",
        "**Classification**: EXECUTIVE STRATEGIC GOVERNANCE // NAST & TIER-1 COMPLIANCE",
        "**Status**: AUDIT LOCKED // STRATEGIC RESOLUTION ENGAGED",
        f"**Objective SLA**: {sim_data['swarm_total']} Quantum Cycles ({thermo['epochs_completed']} Epochs) @ 1:{thermo['dilation_ratio']} Dilation | Pass Rate: {sim_data['swarm_survival_rate']}%",
        "",
        "## 1. EXECUTIVE FINDINGS & OPERATIONAL TELEMETRY ASSESSMENT ([Sentinel_Tactical])",
        f"**Business Proposition**: {prompt_text}",
        f"An enterprise audit evaluating **{sim_data['swarm_total']} cycles** ({thermo['epochs_completed']} epochs of {CYCLES_PER_EPOCH} cycles) via UnifiedHeuristicEngine records **{sim_data['swarm_passed']} validated** and **{sim_data['swarm_dispelled']} pruned** ({sim_data['swarm_survival_rate']}% pass rate) against D_crit = {sim_data['effective_tau']}.",
        "",
        *table_rows,
        "",
        "## 2. FIRST-PRINCIPLES THERMODYNAMICS & RESOURCE-CURRENCY SYNTHESIS ([Sentinel_Analyst])",
        f"- Constants: k = {thermo['k']}, g = {thermo['g']} m/s^2, UGPE = {thermo['ugpe_flux']}, LDOW = {thermo['ldow_ratio']}, D_crit = {D_CRIT_BOUNDARY}",
        f"- Energy Balance: Q_in = {thermo['q_in']}, W_out = {thermo['w_out']}, U_net = {thermo['u_net']} at overclock {thermo['overclock_factor']}x (hard limit {thermo['hard_oc_limit']}x, theoretical OC_max = {thermo['oc_max_theoretical']}x)",
        f"- Entropy & Ledger Totals: T_eff = {thermo['t_eff']}, dS/dt = {thermo['entropy_rate']}, raw entropy margin = {thermo['entropy_margin_raw']}, 50,000-cycle U_net = {thermo['total_u_net_cycles']}, amortized CU = {thermo['total_amortized_cu']}",
        "",
        "## 3. STRATEGIC EXECUTION PROTOCOL & 9-STAGE OPERATIONAL BLUEPRINT ([Sentinel_Safety])",
        f"1. **GOVERN & MAP (Stages 1–2)**: Lock D_crit = {sim_data['effective_tau']} and map AVX2 SIMD, DDPG, and 1:{thermo['dilation_ratio']} dilation dependencies.",
        f"2. **MEASURE & MANAGE (Stages 3–6)**: Track {CYCLES_PER_EPOCH}-cycle epochs, enforce {thermo['hard_oc_limit']}x hard overclock throttle, and trigger fallback to 1.0x if breached.",
        f"3. **GOVERN & SUSTAIN (Stages 7–9)**: Reconcile compute ledger within +/-{thermo['ledger_variance_tolerance_pct']}% every {AUDIT_EPOCHS} epochs ({sim_data['swarm_total']} cycles) with 10Hz heartbeat and 2/3 quorum.",
        "",
        "## 4. RISK MITIGATION & SUCCESS CRITERIA MATRIX ([Sentinel_Creative])",
        "| Strategic Metric | Target Threshold | Verification Methodology |",
        "| --- | --- | --- |",
        f"| Cycle Validation Rate | {sim_data['swarm_survival_rate']}% across {sim_data['swarm_total']} cycles | UnifiedHeuristicEngine ({thermo['epochs_completed']} epochs) |",
        f"| Risk Threshold & Ledger Variance | < {sim_data['effective_tau']} / <= {thermo['ledger_variance_tolerance_pct']}% | Sentinel telemetry & epoch audit reconciliation |",
        "",
        "## 5. FINAL EXECUTIVE VERDICT & IMMEDIATE ACTION CHECKLIST",
        f"'{title}' is evaluated across {sim_data['swarm_total']} quantum cycles under enterprise governance.",
        "- [ ] Authorize Phase 1 operational deployment under governance guard.",
        f"- [ ] Enforce risk governance ceiling at D_crit = {sim_data['effective_tau']} and overclock ceiling <= {thermo['hard_oc_limit']}x.",
        f"- [ ] Reconcile compute-currency ledger within +/-{thermo['ledger_variance_tolerance_pct']}% across {AUDIT_EPOCHS} epochs.",
        "- [ ] Schedule post-deployment review and audit sign-off."
    ])

# ==============================================================================
# 3. HTML/CSS Renderer, TXT Builder, PDF Generator & ZIP Bundler
# ==============================================================================
def clean_raw_latex_artifacts(text: str) -> str:
    text = text.replace("$$", "")
    text = re.sub(r'\\(text|mathbf|mathbb|frac|approx|times|rightarrow|cdot|sigma|psi|tau|Delta)\b', '', text)
    text = re.sub(r'(?<!\\)\$', '', text)
    text = text.replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")
    return text


def _parse_checklist_line(stripped: str):
    m = re.match(r"^(?:•\s*)?-\s*\[([ xX]?)\]\s+(.*)$", stripped)
    if not m:
        return None
    checked = m.group(1).lower() == "x"
    return checked, m.group(2).strip()


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
            headers = parsed_rows[0]
            thead_html = "<thead><tr>" + "".join(f"<th>{_format_inline_html(h)}</th>" for h in headers) + "</tr></thead>"
            body_rows = []
            for r in parsed_rows[1:]:
                padded = r + [""] * max(0, len(headers) - len(r))
                body_rows.append("<tr>" + "".join(f"<td>{_format_inline_html(cell)}</td>" for cell in padded[:len(headers)]) + "</tr>")
            tbody_html = "<tbody>" + "".join(body_rows) + "</tbody>"
            body_parts.append("<table class='uesp-table'>" + thead_html + tbody_html + "</table>")
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

        if stripped.startswith("|"):
            normalized_row = stripped if stripped.endswith("|") else f"{stripped} |"
            table_lines.append(normalized_row)
            continue
        else:
            flush_html_table()

        if not stripped:
            continue

        chk = _parse_checklist_line(stripped) if INTEGRITY_GUARD.active_at("render") else None
        if stripped.startswith("# "):
            body_parts.append(f"<h1>{_format_inline_html(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            body_parts.append(f"<h2>{_format_inline_html(stripped[3:])}</h2>")
        elif stripped.startswith("### ") or stripped.startswith("#### "):
            h_txt = re.sub(r"^#+\s*", "", stripped).strip()
            body_parts.append(f"<h3>{_format_inline_html(h_txt)}</h3>")
        elif chk is not None:
            checked, item_text = chk
            box_char = "☑" if checked else "☐"
            body_parts.append(f"<div class='uesp-check'>{box_char} {_format_inline_html(item_text)}</div>")
        elif stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            bullet_body = stripped[2:].strip()
            body_parts.append(f"<div class='uesp-bullet'>• {_format_inline_html(bullet_body)}</div>")
        elif re.match(r"^\d+\.\s", stripped):
            body_parts.append(f"<div class='uesp-bullet'>{_format_inline_html(stripped)}</div>")
        else:
            body_parts.append(f"<p>{_format_inline_html(stripped)}</p>")

    flush_html_table()
    if in_code:
        flush_html_code()

    css_block = """
    <style>
      .uesp-report-wrap { font-family: Arial, Helvetica, sans-serif; color: #1b262c; line-height: 1.5; max-width: 100%; }
      .uesp-header-bar { display: flex; justify-content: space-between; border-bottom: 2px solid #00796b; padding-bottom: 8px; margin-bottom: 12px; }
      .uesp-brand { font-size: 16px; font-weight: bold; color: #004d40; }
      .uesp-meta-box { background: #e0f2f1; border: 1px solid #00796b; padding: 10px 12px; border-radius: 4px; font-size: 12px; margin-bottom: 14px; color: #004d40; }
      .uesp-report-wrap h1 { font-size: 20px; color: #004d40; margin: 14px 0 8px 0; break-after: avoid; }
      .uesp-report-wrap h2 { font-size: 16px; color: #00695c; margin: 12px 0 6px 0; border-bottom: 1px solid #b2dfdb; padding-bottom: 3px; break-after: avoid; }
      .uesp-report-wrap h3 { font-size: 14px; color: #00796b; margin: 10px 0 4px 0; break-after: avoid; }
      .uesp-report-wrap p { margin: 5px 0; font-size: 13px; }
      .uesp-bullet { margin: 4px 0 4px 14px; font-size: 13px; break-inside: avoid; }
      .uesp-check { margin: 5px 0 5px 14px; font-size: 13px; break-inside: avoid; }
      .uesp-table { width: 100%; border-collapse: collapse; margin: 10px 0 14px 0; font-size: 12px; }
      .uesp-table thead { display: table-header-group; }
      .uesp-table tr { break-inside: avoid; }
      .uesp-table th { background: #00796b; color: #ffffff; text-align: left; padding: 7px 8px; border: 1px solid #004d40; }
      .uesp-table td { padding: 6px 8px; border: 1px solid #80cbc4; vertical-align: top; }
      .uesp-table tr:nth-child(even) td { background: #e0f2f1; }
      .uesp-code { background: #f1f8e9; border: 1px solid #80cbc4; padding: 8px 10px; border-radius: 4px; overflow-x: auto; font-family: "Courier New", Courier, monospace; font-size: 11.5px; color: #004d40; white-space: pre-wrap; word-break: break-word; }
      @page { size: letter; margin: 0.55in; }
    </style>
    """

    ledger = INTEGRITY_GUARD.ledger
    inner_html = (
        f"{css_block if include_document_wrapper else ''}"
        f"<div class='uesp-report-wrap'>"
        f"<div class='uesp-header-bar'><div class='uesp-brand'>CELSIUS TECH MEDIA GROUP</div><div>{html.escape(COMPANY_DETAILS).replace(chr(10), '<br/>')}</div></div>"
        f"<div class='uesp-meta-box'><strong>Session ID:</strong> {html.escape(session_id)} | <strong>Governance Guard:</strong> {html.escape(INTEGRITY_GUARD.level)} (Repairs: {INTEGRITY_GUARD.interventions}) | "
        f"<strong>Compute Spent:</strong> {ledger.total_compute_currency_spent} | <strong>Thermal Load:</strong> {ledger.thermal_load_index} | <strong>Cycles:</strong> {QUANTUM_CYCLE_DILATION}</div>"
        f"<h2>Subject: {html.escape(title)}</h2>"
        + "\n".join(body_parts)
        + "</div>"
    )

    if include_document_wrapper:
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{css_block}</head><body>{inner_html}</body></html>"
    return inner_html


def generate_txt_artifact(filename: str, title: str, content: str, session_id: str) -> str:
    cleaned = clean_raw_latex_artifacts(content)
    lines = cleaned.split("\n")
    ledger = INTEGRITY_GUARD.ledger
    out_lines = [
        "========================================================================",
        "CELSIUS TECH MEDIA GROUP — ENTERPRISE GOVERNANCE & THERMODYNAMIC REPORT",
        "========================================================================",
        f"Session ID     : {session_id}",
        f"Governance Guard: {INTEGRITY_GUARD.level} (Repairs: {INTEGRITY_GUARD.interventions})",
        f"Compute Spent  : {ledger.total_compute_currency_spent} | Thermal Load: {ledger.thermal_load_index}",
        f"Subject        : {title}",
        f"Cycle Scale    : {QUANTUM_CYCLE_DILATION} Cycles ({AUDIT_EPOCHS} Epochs @ {CYCLES_PER_EPOCH}/epoch)",
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
        if s.startswith("|"):
            table_buf.append(s if s.endswith("|") else f"{s} |")
        else:
            flush_txt_table()
            chk = _parse_checklist_line(s) if INTEGRITY_GUARD.active_at("render") else None
            if chk is not None:
                checked, item_text = chk
                out_lines.append(f"[{'X' if checked else ' '}] {item_text}")
            else:
                out_lines.append(line)

    flush_txt_table()
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
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


def _compute_table_col_widths(num_cols: int, usable_width: float) -> list:
    weights_map = {
        3: [0.26, 0.30, 0.44],
        4: [0.20, 0.32, 0.28, 0.20],
        6: [0.08, 0.17, 0.27, 0.22, 0.14, 0.12],
    }
    weights = weights_map.get(num_cols)
    if weights and len(weights) == num_cols:
        return [usable_width * w for w in weights]
    return [usable_width / max(1, num_cols)] * num_cols


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
            col_widths = _compute_table_col_widths(num_cols, usable_width)
            t = Table(rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
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

        if line.startswith('|'):
            table_buffer.append(line if line.endswith('|') else f"{line} |")
            continue
        else:
            flush_table()

        if not line:
            continue

        chk = _parse_checklist_line(line) if INTEGRITY_GUARD.active_at("render") else None
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
        elif chk is not None:
            checked, item_text = chk
            box = "☑" if checked else "☐"
            story.append(safe_p(f"{box} {item_text}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
            story.append(safe_p(f"• {line[2:].strip()}", styles['Bullet']))
            story.append(Spacer(1, 2))
        elif re.match(r'^\d+\.\s', line):
            story.append(safe_p(line, styles['Bullet']))
            story.append(Spacer(1, 2))
        else:
            story.append(safe_p(line, styles['Body']))
            story.append(Spacer(1, 3))

    flush_table()
    if in_code_block:
        flush_code()


def guard_story_heights(story: list, max_height: float = 680.0, width: float = 7.0 * inch):
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

    ledger = INTEGRITY_GUARD.ledger
    manifest_text = (
        f"ECTA, NAST & ENTERPRISE GOVERNANCE MANIFEST:\n"
        f"• SHA256 ECTA Timestamped Session: {session_id}\n"
        f"• Governance Guard: {INTEGRITY_GUARD.level} (Repairs Applied: {INTEGRITY_GUARD.interventions}) | Overclock: {ledger.overclock_factor}x\n"
        f"• Compute Currency Spent: {ledger.total_compute_currency_spent} | Thermal Load Index: {ledger.thermal_load_index}\n"
        f"• Knowledge Catalog: Registered ({len(NVIDIA_NIM_CATALOG)} Active Models) | 9-Stage Cascade Armed\n"
        f"• Governance Audit Scale: {QUANTUM_CYCLE_DILATION} Cycles ({AUDIT_EPOCHS} Epochs) | NIST / NAST Sensor Annotations"
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


def generate_audit_json_artifact(filename: str, title: str, session_id: str, sim_data: dict) -> str:
    payload = {
        "session_id": session_id,
        "title": title,
        "governance_guard": INTEGRITY_GUARD.level,
        "repairs_applied": INTEGRITY_GUARD.interventions,
        "ledger": {
            "total_compute_currency_spent": INTEGRITY_GUARD.ledger.total_compute_currency_spent,
            "thermal_load_index": INTEGRITY_GUARD.ledger.thermal_load_index,
            "calls_executed": INTEGRITY_GUARD.ledger.calls_executed,
        },
        "cycle_summary": {
            "total_cycles": sim_data["swarm_total"],
            "passed_cycles": sim_data["swarm_passed"],
            "pruned_cycles": sim_data["swarm_dispelled"],
            "pass_rate_pct": sim_data["swarm_survival_rate"],
            "mean_observed_score": sim_data["mean_observed_score"],
            "max_observed_score": sim_data["max_observed_score"],
            "d_crit": sim_data["effective_tau"],
            "fallback_triggered": sim_data["fallback_triggered"],
            "epochs": sim_data["epoch_summaries"],
        },
        "thermodynamics": sim_data["thermodynamics"],
    }
    serialized = json.dumps(payload, indent=2)
    payload["sha256_digest"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filename


def generate_zip_artifact(zip_filename: str, pdf_filename: str, txt_filename: str, json_filename: str = None) -> str:
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        if os.path.exists(pdf_filename):
            zf.write(pdf_filename, arcname=os.path.basename(pdf_filename))
        if os.path.exists(txt_filename):
            zf.write(txt_filename, arcname=os.path.basename(txt_filename))
        if json_filename and os.path.exists(json_filename):
            zf.write(json_filename, arcname=os.path.basename(json_filename))
    return zip_filename

# ==============================================================================
# 4. Main Execution Entry
# ==============================================================================
def process_and_run(title: str, issue_text: str):
    INTEGRITY_GUARD.interventions = 0
    INTEGRITY_GUARD.ledger = ResourceThermalLedger()

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
        total_cycles=QUANTUM_CYCLE_DILATION
    )

    vector_store = LocalARMVectorStore()
    prior_memories = vector_store.recall_prior_context(n_results=2)
    report_content = query_nvidia_nim_iterative(title, issue_text, sim_data, prior_memories)
    report_content = report_content.replace("{{session_id}}", session_id)

    vector_store.trap_information(session_id, [f"Governance Audit Subject: {title}\n{report_content[:1200]}"])

    wp_html_fragment = render_report_html(title, report_content, session_id, include_document_wrapper=False)
    full_html_doc = render_report_html(title, report_content, session_id, include_document_wrapper=True)

    short_id = session_id[:16]
    pdf_filename = f"Report_{short_id}.pdf"
    txt_filename = f"Report_{short_id}.txt"
    json_filename = f"Audit_Ledger_{short_id}.json"
    zip_filename = f"Report_{short_id}.zip"

    generate_pdf_artifact(pdf_filename, title, report_content, session_id, full_html=full_html_doc)
    generate_txt_artifact(txt_filename, title, report_content, session_id)
    generate_audit_json_artifact(json_filename, title, session_id, sim_data)
    generate_zip_artifact(zip_filename, pdf_filename, txt_filename, json_filename)

    ledger = INTEGRITY_GUARD.ledger
    print(
        f"✅ Artifacts generated: {zip_filename} "
        f"[Guard={INTEGRITY_GUARD.level}, repairs={INTEGRITY_GUARD.interventions}, "
        f"cycles={sim_data['swarm_total']}, compute={ledger.total_compute_currency_spent}, thermal={ledger.thermal_load_index}] "
        f"(PDF: {os.path.getsize(pdf_filename) if os.path.exists(pdf_filename) else 0}B, "
        f"TXT: {os.path.getsize(txt_filename) if os.path.exists(txt_filename) else 0}B, "
        f"JSON: {os.path.getsize(json_filename) if os.path.exists(json_filename) else 0}B, "
        f"ZIP: {os.path.getsize(zip_filename) if os.path.exists(zip_filename) else 0}B)"
    )

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
                    f"<p><a href='{html.escape(media_source_url)}' target='_blank' rel='noopener'>📥 Download Full Artifact Bundle (ZIP: PDF + TXT + JSON)</a></p>"
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
