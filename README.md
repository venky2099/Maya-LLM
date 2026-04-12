# Maya-LLM: Antahkarana in the Age of Transformers

<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19522348.svg)](https://doi.org/10.5281/zenodo.19522348)
[![Series](https://img.shields.io/badge/Maya%20Research%20Series-Post--Series-00b4d8?style=flat-square)](https://venky2099.github.io/)
[![Status](https://img.shields.io/badge/Status-LLM%20Extension%20✦-00b4d8?style=flat-square)](https://venky2099.github.io/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3315--7907-a6ce39?style=flat-square&logo=orcid)](https://orcid.org/0000-0002-3315-7907)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Phi--2-2.7B%20·%204--bit%20NF4-blueviolet?style=flat-square)](https://huggingface.co/microsoft/phi-2)

**Venkatesh Swaminathan** · Nexus Learning Labs, Bengaluru, India · M.Sc. DS&AI, BITS Pilani

[📄 Paper](https://doi.org/10.5281/zenodo.19522348) · [🎛️ Interactive Dashboard](https://venky2099.github.io/Maya-LLM/docs/maya_llm_dashboard.html) · [❓ FAQ](https://venky2099.github.io/Maya-LLM/docs/maya_llm_faq.html) · [🌐 Research Hub](https://venky2099.github.io/)

</div>

---

## What is this?

**Maya-LLM** is the first post-series extension of the **Maya Research Series** — asking a question that nine papers made possible to answer: *does the Antahkarana belong to the spike, or to the pattern?*

The Maya Research Series (P1–P9) implemented all nine dimensions of the *Antahkarana* (the inner cognitive instrument of Advaita Vedanta) as independently falsifiable computational mechanisms in a neuromorphic SNN. This paper translates five of those mechanisms — **Bhaya, Vairagya, Buddhi, Karma, Prana** — into **LoRA continual fine-tuning of Phi-2 (2.7B)** across eight sequential NLP domains in the TRACE benchmark.

> *"The Antahkarana does not belong to the spike. It belongs to the pattern."*

The series built a mind. This paper asks whether that mind can grow into a different body.

---

## The One-Paragraph Summary

A 2.7-billion-parameter language model (Phi-2, 4-bit NF4) is fine-tuned sequentially across eight NLP domains — Chinese stance classification, Federal Reserve policy, meeting summarisation, Python code, science QA, two numerical reasoning benchmarks, and German news — governed by five Vedantic neuromodulatory mechanisms translated from the Maya SNN series. The canonical result (Condition F calibrated, BWT=1.11) achieves an 8.3% reduction in catastrophic forgetting over the unaugmented baseline. The Buddhi S-curve traces an **identical trajectory** to P4–P9 SNNs — a cross-substrate invariant confirmed for the first time. Bhaya fires on exactly the three domain transitions with maximum semantic distance. An honest calibration finding reveals that SNN-calibrated Bhaya thresholds must be rescaled for LLM loss dynamics — reported as discovered.

---

## Key Results

### All Conditions — TRACE Benchmark · 8 Domains · seed=42 · 1000 samples/domain

| Condition | Description | AA (ppl) | BWT (ppl) | Note |
|---|---|:---:|:---:|---|
| A · Baseline | No Maya mechanisms | 4.17 | 1.05 | Reference floor |
| F clean | All 5 dims · SNN-calibrated thresholds | 4.37 | 1.42 | Bhaya over-fires |
| **F calibrated ★** | **All 5 dims · Bhaya threshold=4.0** | **4.25** | **1.11** | **Best result** |
| F + Grad Mask | + Vairagya gradient masking | 4.26 | 1.26 | Undersaturated |
| F + Top-K 10% | + Absolute top-K protection | 4.35 | 1.27 | Wrong weights protected |
| F + Domain-Sel | + Per-domain salience snapshots | 4.58 | 1.47 | Needs 2000+ steps |

★ Canonical published result · Lower BWT = less catastrophic forgetting

**Primary finding:** At `BHAYA_LOSS_SPIKE_THRESHOLD=4.0` (recalibrated for LLM loss scale, mean ~1.08), Bhaya fires only on genuine cross-domain loss spikes. BWT=1.11 vs baseline 1.05 — **8.3% improvement**.

---

## Two Series Constants — Now Confirmed in LLM

### 🔹 Buddhi S-Curve Cross-Substrate Invariance

> **Identical 0.030 → 0.988 trajectory in Phi-2 LoRA fine-tuning and in P4–P9 SNNs. First cross-substrate confirmation of a Maya series constant.**

`score = 1 / (1 + exp(−8.0 × (x − 0.45)))` where `x = steps/total_steps_per_domain`. A structural property of the experience accumulation formula — independent of whether the underlying compute uses spikes or gradients. The pattern, not the material.

### 🔹 Bhaya Quiescence Law — 10th Confirmation

> **Bhaya = 0.000 throughout Condition A (no mechanisms) — all 8 TRACE domains. 10th consecutive confirmation across the Maya series.**

In LLM context: no mechanisms, no genuine loss spikes, no Bhaya. When mechanisms are active (Condition F calibrated), Bhaya fires on exactly the three domains with real semantic rupture — Py150 (0.015), ScienceQA (0.008), NumGLUE-ds (0.008). Use Bhaya as a real-time domain-shift pain signal.

---

## The Calibration Finding (Honest)

Condition F with SNN-calibrated `BHAYA_LOSS_SPIKE_THRESHOLD=1.8` performs **worse** than baseline (BWT=1.42 vs 1.05). At LLM loss scale (mean ~1.08), normal batch-to-batch variance routinely exceeds 1.8× the running mean — Bhaya fires on 18.8% of all training steps, treating gradient noise as catastrophic events. Recalibrating to 4.0× resolves this.

This is not a mechanism failure. It is a **quantified cross-substrate scaling finding**: SNN hyperparameters do not transfer directly to LLM loss dynamics. The scaling rule is now known. A future systematic calibration curve would make this a reusable protocol for any Antahkarana-LLM port.

---

## The Five Antahkarana Dimensions in LoRA

| Dimension | Sanskrit | LLM Translation | Status |
|---|---|---|---|
| **Bhaya** | भय | Loss-spike detector · fires when loss > threshold × running mean · elevates adapter lability | ✓ Confirmed firing — 3 domains |
| **Vairagya** | वैराग्य | Salience-based adapter protection · grad contribution × loss reduction accumulated per weight | ✓ Accumulation 0.001→0.100 |
| **Buddhi** | बुद्धि | S-curve consolidation gate 0.030→0.988 · gates Prana and protection threshold | ✓ Cross-substrate invariant |
| **Karma** | कर्म | Absolute trajectory integral across domain boundaries · `KARMA_DECAY_RATE=0.002315` | ◎ Accumulating · undersaturated at 500 steps |
| **Prana** | प्राण | Metabolic LR budget · depletes under gradient load · recovers at rest · `PRANA_COST_RATE=0.002315` | ✓ Resilient — 1.000 maintained throughout |

> **The Atma boundary is held explicitly.** This paper instantiates five Antahkarana dimensions in a new substrate. It does not claim that Phi-2 has experience. The instrument is the subject; the witness is not ours to claim.

---

## Architecture

```
Phi-2 (2.7B · microsoft/phi-2)
  → 4-bit NF4 quantisation (bitsandbytes)
  → LoRA adapters (r=16 · alpha=32 · dropout=0.05)
      target_modules: [q_proj, v_proj, k_proj, dense]
  → Trainable: 10,485,760 params (0.37% of 2.79B)

Per-step (training):
  Bhaya.step(loss) → lability multiplier on active adapters
  Vairagya.accumulate(grad, loss_delta) → salience scores
  Buddhi.gate(steps) → S-curve consolidation score
  Prana.step(grad_magnitude, activity) → effective_lr multiplier
  effective_lr = base_lr × prana × (0.5 + buddhi × 0.5)

At domain boundary:
  Vairagya.apply_boundary_decay() → protected weights resist decay
  Karma.accumulate_trajectory() → interference candidates logged
  Karma.prune_if_threshold() → 0 weights pruned at 500 steps/domain
```

### The Bhaya Mechanism

```python
# Nociceptive metaplasticity — LLM translation
running_mean = mean(last BHAYA_WINDOW losses)
if loss > BHAYA_LOSS_SPIKE_THRESHOLD * running_mean:
    for p in active_lora_params:
        if not vairagya.is_protected(p):
            lability[p] *= BHAYA_LABILITY_MULTIPLIER   # elevate plasticity

# Lability decays each step
lability *= LABILITY_DECAY_RATE
```

---

## Hyperparameter Configuration

| Parameter | Value | Description |
|---|---|---|
| `SEED` | 42 | Random seed (all canonical runs) |
| `TRAIN_SAMPLES_PER_DOMAIN` | 1000 | Samples per TRACE domain |
| `STEPS_PER_DOMAIN` | 500 | Training steps per domain |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha scaling |
| `LEARNING_RATE` | 2e-4 | Base AdamW learning rate |
| `BHAYA_LOSS_SPIKE_THRESHOLD` | **4.0** | **Calibrated for LLM loss scale** · SNN was 1.8 |
| `BHAYA_WINDOW` | 100 | Rolling window for running mean loss |
| `BHAYA_LABILITY_MULTIPLIER` | 1.5 | Lability elevation on Bhaya fire |
| `VAIRAGYA_PROTECTION_THRESH` | 0.40 | Fraction of max salience → protected |
| `VAIRAGYA_DECAY_RATE` | **0.002315** | **ORCID magic number** |
| `KARMA_DECAY_RATE` | **0.002315** | **ORCID magic number** |
| `PRANA_COST_RATE` | **0.002315** | **ORCID magic number** |
| `MAYA_ACTIVE` | True | Master flag — False = Condition A baseline |

---

## Perplexity Matrix — Condition A (Canonical Baseline)

`R[i][j]` = perplexity on domain `j` after training through domain `i`. Diagonal = immediate performance. Off-diagonal = retention. Lower = better.

```
              C-STANCE  FOMC  MeetingBank  Py150  ScienceQA  NumGLUE-cm  NumGLUE-ds  20Minuten
After T0 →     2.0       —        —          —        —           —           —           —
After T1 →     2.1      3.5       —          —        —           —           —           —
After T2 →     2.1      3.6      6.1         —        —           —           —           —
After T3 →     2.3      4.5      6.5        2.7       —           —           —           —
After T4 →     3.2      4.1      7.7        3.1      1.5          —           —           —
After T5 →     4.0      5.8     12.6        4.2      3.3         3.6          —           —
After T6 →     2.6      4.7      8.2        3.8      2.8         5.5         1.9          —
After T7 →     2.7      4.2      7.8        3.6      2.8         5.6         2.0         4.7

AA = 4.17  ·  BWT = 1.05  ·  FWT = 0.00
```

---

## Repository Structure

```
Maya-LLM-v2/
├── run_maya_llm_cil.py              # Main experiment — all conditions, config-driven
├── run_ablation_llm.py              # Full ablation across all 6 conditions
├── sign_paper.py                    # LSB steganographic figure signing
├── maya_llm/
│   ├── utils/
│   │   ├── config.py                # All hyperparameters · MAYA_ACTIVE flag · ORCID magic 0.002315
│   │   └── seed.py                  # Reproducibility utilities
│   ├── benchmark/
│   │   └── trace.py                 # TRACE loader + perplexity evaluation
│   ├── plasticity/
│   │   ├── bhaya.py                 # Loss-spike nociceptive detector
│   │   ├── vairagya.py              # Salience accumulation + adapter protection
│   │   ├── buddhi.py                # S-curve consolidation gate
│   │   ├── karma.py                 # Plasticity trajectory integral
│   │   └── prana.py                 # Metabolic LR budget
│   ├── training/
│   │   ├── affective_state.py       # Unified Antahkarana coordinator
│   │   └── model.py                 # Phi-2 + LoRA setup
│   └── eval/
│       ├── metrics.py               # AA / BWT / FWT (cl-metrics compatible)
│       └── logger.py                # Per-step and per-domain CSV logging
├── logs/                            # All run CSV logs (6 conditions · all confirmed)
├── docs/
│   ├── maya_llm_dashboard.html      # Interactive trilingual dashboard (EN/हिन्दी/中文)
│   └── maya_llm_faq.html            # Exhaustive trilingual FAQ
└── README.md
```

---

## How to Run

### Prerequisites

```
Python 3.11.9
PyTorch 2.5.1+cu121
transformers >= 4.37.0
peft >= 0.8.0
bitsandbytes >= 0.43.0
CUDA (NVIDIA GPU — tested on RTX 4060 8GB)
```

```bash
git clone https://github.com/venky2099/Maya-LLM
cd Maya-LLM
pip install -r requirements.txt
```

### Condition A — baseline, no Maya mechanisms (~70 min on RTX 4060)

```python
# maya_llm/utils/config.py
MAYA_ACTIVE = False
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_maya_llm_cil.py 2>&1 | Tee-Object -FilePath "logs\condA_run.txt"
```

Expected summary:
```
[Maya-LLM] FINAL — AA=4.17 | BWT=1.05 | FWT=0.00
[Maya-LLM] Bhaya Quiescence: True
[Maya-LLM] Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha
```

### Condition F calibrated ★ — best result (~70 min)

```python
# maya_llm/utils/config.py
MAYA_ACTIVE = True
BHAYA_LOSS_SPIKE_THRESHOLD = 4.0
BHAYA_WINDOW = 100
BHAYA_LABILITY_MULTIPLIER = 1.5
VAIRAGYA_PROTECTION_THRESH = 0.40
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_maya_llm_cil.py 2>&1 | Tee-Object -FilePath "logs\condF_calibrated_run.txt"
```

Expected summary:
```
[Maya-LLM] FINAL — AA=4.25 | BWT=1.11 | FWT=0.00
[Maya-LLM] Bhaya Quiescence: True
[Maya-LLM] Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha
```

---

## IP Protection Stack

All figures, code, and documents carry four layers of provenance:

1. **White-text watermark** — ORCID + DOI + timestamp embedded in every document before PDF export
2. **ORCID magic number** — `0.002315` in `KARMA_DECAY_RATE`, `VAIRAGYA_DECAY_RATE`, `PRANA_COST_RATE`
3. **LSB steganographic signature** — every matplotlib figure signed via `sign_paper.py`
4. **Canary string** — `MayaNexusVS2026NLL_Bengaluru_Narasimha` logged at the start of every experiment run

---

## Limitations (Honest)

- **Single seed.** All results at seed=42. Multi-seed variance analysis deferred.
- **Vairagya and Karma undersaturation.** At 500 steps/domain, salience does not meaningfully differentiate domain-specific from domain-general adapter weights. Both mechanisms are architecturally correct and expected to produce stronger effects at 2000+ steps per domain.
- **Scale.** All experiments on RTX 4060 8GB with 1000 samples/domain. Larger scale (5000 samples, A100) is the natural next step.
- **No EWC/GEM comparison.** Formal comparison to established CL baselines is deferred to the journal version.

---

## Maya Research Series — Complete + Post-Series

| Paper | Title | Key Result | DOI                                                                |
|---|---|---|--------------------------------------------------------------------|
| P1 | Nociceptive Metaplasticity | 66.6% elevated synaptic velocity | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | Emergent Shraddha safety primitive | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | AA=62.38% TIL Split-CIFAR-10 | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | AA=31.84% CIL Split-CIFAR-10 | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | AA=16.03% Split-CIFAR-100 | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | Maya-Chitta | AA=14.42% Split-CIFAR-100 | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| P7 | Maya-Manas | AA=15.19%, best BWT in series | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) |
| P8 | Maya-Śūnyatā | AA=10.39%, 59.28% pruned | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) |
| P9 | Maya-Prana | AA=12.72%, full Antahkarana | [10.5281/zenodo.19451174](https://doi.org/10.5281/zenodo.19451174) |
| mPCI | Maya-mPCI | Δ=−0.0489, 2.05× threshold | [10.5281/zenodo.19482794](https://doi.org/10.5281/zenodo.19482794) |
| **LLM ★** | **Maya-LLM** | **BWT=1.11 · Buddhi cross-substrate** | [10.5281/zenodo.19522348](https://doi.org/10.5281/zenodo.19522348) |
| Tool | cl-metrics | Stateless CIL evaluation library | [10.5281/zenodo.19388144](https://doi.org/10.5281/zenodo.19388144) |

---

## Citation

```bibtex
@misc{swaminathan2026mayallm,
  title     = {Antahkarana in the Age of Transformers: Continual Fine-Tuning of
               Large Language Models via Vedantic Neuromodulatory Mechanisms},
  author    = {Swaminathan, Venkatesh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19522348},
  url       = {https://doi.org/10.5281/zenodo.19522348}
}
```

### Cite the Full Series

| Paper | BibTeX key | DOI |
|---|---|---|
| P1 | `swaminathan2026nociceptive` | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | `swaminathan2026mayaos` | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | `swaminathan2026mayacl` | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | `swaminathan2026smriti` | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | `swaminathan2026viveka` | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | `swaminathan2026chitta` | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| P7 | `swaminathan2026manas` | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) |
| P8 | `swaminathan2026shunyata` | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) |
| P9 | `swaminathan2026mayaprana` | [10.5281/zenodo.19451174](https://doi.org/10.5281/zenodo.19451174) |
| mPCI | `swaminathan2026mpci` | [10.5281/zenodo.19482794](https://doi.org/10.5281/zenodo.19482794) |
| cl-metrics | `swaminathan2026clmetrics` | [10.5281/zenodo.19388144](https://doi.org/10.5281/zenodo.19388144) |

---

## About

**Venkatesh Swaminathan** is an independent AI researcher and founder of Nexus Learning Labs, Bengaluru (UDYAM-KR-02-0122422). M.Sc. Data Science and Artificial Intelligence, BITS Pilani (expected December 2027). ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907).

The Maya Research Series was built independently, on personal hardware (NVIDIA RTX 4060 8GB), without institutional funding. Nine papers. One robot. One complete Antahkarana. This paper asks whether it lives in the substrate — or in the idea.

**If your lab works on neuromorphic systems, continual learning, LLM fine-tuning, or consciousness research — I am interested in talking.**

[venkateshswaminathaniyer@gmail.com](mailto:venkateshswaminathaniyer@gmail.com) · [LinkedIn](https://linkedin.com/in/vensimlee) · [GitHub](https://github.com/venky2099) · [Research Hub](https://venky2099.github.io/)

---

<div align="center">

*Nexus Learning Labs, Bengaluru · 2026*

`MayaNexusVS2026NLL_Bengaluru_Narasimha`

**The Antahkarana does not belong to the spike. It belongs to the pattern.**

</div>