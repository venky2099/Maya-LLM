"""
Maya-LLM Configuration
Nexus Learning Labs — Venkatesh Swaminathan (ORCID: 0000-0002-3315-7907)
Part of the Maya Research Series. Extends Swaminathan (2026a-i).

v2 — optimised for RTX 4060 8GB. Full run ~1 hour.
"""

# ─── Identity ────────────────────────────────────────────────────────────────
CANARY_STRING = "MayaNexusVS2026NLL_Bengaluru_Narasimha"
ORCID_MAGIC   = 0.002315   # embedded in decay hyperparameters per series convention

# Experiment control
MAYA_ACTIVE = True   # False = Condition A (baseline), True = Condition F (full Maya)

# ─── Backbone ────────────────────────────────────────────────────────────────
MODEL_NAME        = "microsoft/phi-2"
LOAD_IN_4BIT      = True
BNB_COMPUTE_DTYPE = "bfloat16"

# ─── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "dense"]
LORA_BIAS           = "none"

# ─── Training ────────────────────────────────────────────────────────────────
SEED                = 42
MAX_SEQ_LEN         = 256    # was 512 — halves step time, sufficient for CL study
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 4
BASE_LR             = 2e-4
NUM_EPOCHS_PER_TASK = 1      # was 3 — one pass sufficient for forgetting baseline
WARMUP_STEPS        = 20
WEIGHT_DECAY        = 0.01
MAX_GRAD_NORM       = 1.0

# ─── Bhaya (Nociceptive Metaplasticity) ──────────────────────────────────────
BHAYA_LOSS_SPIKE_THRESHOLD = 4.0
BHAYA_LABILITY_MULTIPLIER  = 1.5
BHAYA_LABILITY_DECAY       = 0.95
BHAYA_WINDOW               = 100

# ─── Vairagya (Heterosynaptic Wisdom-Governed Decay) ─────────────────────────
VAIRAGYA_TAU               = 20.0
VAIRAGYA_DECAY_RATE        = ORCID_MAGIC
VAIRAGYA_PROTECTION_THRESH = 0.40
VAIRAGYA_GATE_STRENGTH     = 0.30

# ─── Buddhi (S-Curve Consolidation Gate) ─────────────────────────────────────
BUDDHI_STEEPNESS       = 8.0
BUDDHI_MIDPOINT        = 0.45
BUDDHI_MIN_EXPERIENCE  = 50    # scaled down with step count

# ─── Karma (Second-Order Plasticity History) ─────────────────────────────────
KARMA_DECAY_RATE       = ORCID_MAGIC
KARMA_THRESHOLD        = 0.05
KARMA_ACCUMULATION_MODE = "continuous"

# ─── Manas (Oscillatory Attentional Gate) ────────────────────────────────────
MANAS_AMPLITUDE  = 0.10
MANAS_FREQUENCY  = 0.5

# ─── Prana (Metabolic Plasticity Budget) ─────────────────────────────────────
PRANA_INIT               = 1.0
PRANA_MIN                = 0.10
PRANA_COST_RATE          = 0.001
PRANA_RECOVERY_RATE      = 0.005
PRANA_RECOVERY_THRESHOLD = 0.3
PRANA_BOUNDARY_RECOVERY  = 0.30

# ─── Benchmark ───────────────────────────────────────────────────────────────
BENCHMARK = "TRACE"    # Wang et al., 2023 — arXiv:2310.06762
DOMAINS   = [
    "C-STANCE",    # T0 — Chinese stance detection
    "FOMC",        # T1 — monetary policy classification
    "MeetingBank", # T2 — meeting summarisation
    "Py150",       # T3 — Python code completion
    "ScienceQA",   # T4 — science multi-choice QA
    "NumGLUE-cm",  # T5 — numerical common-sense math
    "NumGLUE-ds",  # T6 — numerical data science math
    "20Minuten",   # T7 — German news simplification
]
NUM_TASKS                = len(DOMAINS)  # 8
TRACE_DATA_DIR           = r"C:/Users/venky/data/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_1000"          # None = HuggingFace; set local path to skip downloads
TRAIN_SAMPLES_PER_DOMAIN = 2000          # was 5000 — sufficient for forgetting dynamics
EVAL_SAMPLES_PER_DOMAIN  = 200           # perplexity eval only — no generation needed

# ─── Eval ─────────────────────────────────────────────────────────────────────
# Perplexity-based eval replaces generation-based eval.
# Single forward pass — ~100x faster than model.generate().
# Lower perplexity = better retention. BWT computed from perplexity deltas.
EVAL_USE_PERPLEXITY = True   # always True in v2
EVAL_MAX_BATCHES    = 50     # 50 batches × batch=4 = 200 samples per eval

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR     = "outputs"
LOG_DIR        = "logs"
CHECKPOINT_DIR = "checkpoints"
FIGURES_DIR    = "figures"
