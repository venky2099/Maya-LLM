"""
run_maya_llm_cil.py — Maya-LLM Continual Fine-Tuning Entry Point v2
Nexus Learning Labs — Venkatesh Swaminathan (ORCID: 0000-0002-3315-7907)
Part of the Maya Research Series.

v2 changes:
  - Perplexity-based eval (no model.generate()) — ~100x faster
  - MAX_SEQ_LEN=256, 1000 samples/domain, 1 epoch
  - Per-domain timing printed
  - Estimated completion time shown at start

Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha
"""

import os
import time
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from maya_llm.utils.config import (
    CANARY_STRING, SEED, BASE_LR, NUM_EPOCHS_PER_TASK,
    WARMUP_STEPS, WEIGHT_DECAY, MAX_GRAD_NORM, DOMAINS, OUTPUT_DIR,
    TRACE_DATA_DIR, TRAIN_SAMPLES_PER_DOMAIN, EVAL_SAMPLES_PER_DOMAIN,
    EVAL_MAX_BATCHES, BATCH_SIZE,
)
from maya_llm.utils.seed import set_seed
from maya_llm.training.model import load_model_and_tokenizer, get_lora_params
from maya_llm.training.affective_state import MayaAffectiveState
from maya_llm.benchmark.trace import load_trace_domain, evaluate_domain_perplexity
from maya_llm.eval.metrics import CLMetrics
from maya_llm.eval.logger import MayaLLMLogger

print(f"[Maya-LLM] Canary: {CANARY_STRING}")
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_eval_loaders = {}


def main():
    run_start = time.time()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Maya-LLM] Device: {device}")

    # ── Pre-run estimate ──────────────────────────────────────────────────────
    steps_per_domain  = (TRAIN_SAMPLES_PER_DOMAIN // BATCH_SIZE) * NUM_EPOCHS_PER_TASK
    total_train_steps = steps_per_domain * len(DOMAINS)
    print(f"[Maya-LLM] Steps/domain: {steps_per_domain} | Total: {total_train_steps}")
    print(f"[Maya-LLM] Estimated runtime: ~{total_train_steps * 1.5 / 3600:.1f}h")
    print(f"[Maya-LLM] Eval: perplexity-based (no generation)")

    model, tokenizer = load_model_and_tokenizer()
    lora_params      = get_lora_params(model)
    print(f"[Maya-LLM] LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    metrics = CLMetrics(num_domains=len(DOMAINS))
    logger  = MayaLLMLogger(run_name="maya_llm_cil_condF_domainSel_s42")

    for domain_idx, domain in enumerate(DOMAINS):
        domain_start = time.time()
        print(f"\n[Maya-LLM] ── Domain {domain_idx}: {domain} ──")

        dataloader, eval_loader = load_trace_domain(
            domain=domain,
            tokenizer=tokenizer,
            data_dir=TRACE_DATA_DIR,
            train_samples=TRAIN_SAMPLES_PER_DOMAIN,
            test_samples=EVAL_SAMPLES_PER_DOMAIN,
            batch_size=BATCH_SIZE,
        )
        _eval_loaders[domain_idx] = eval_loader

        affective = MayaAffectiveState(lora_params, steps_per_domain)
        optimizer = AdamW(lora_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=steps_per_domain,
        )

        model.train()
        global_step = 0

        for epoch in range(NUM_EPOCHS_PER_TASK):
            for batch in dataloader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, MAX_GRAD_NORM)

                affective_signals = affective.step(loss.item(), optimizer)

                # Prana gates effective LR
                for group in optimizer.param_groups:
                    group["lr"] = BASE_LR * affective_signals["prana_level"]

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                logger.log_step(global_step, domain_idx, loss.item(), affective_signals)
                global_step += 1

                if global_step % 50 == 0:
                    elapsed = time.time() - domain_start
                    sps     = global_step / max(elapsed, 1)  # steps per second
                    eta_s   = (steps_per_domain - global_step) / max(sps, 0.01)
                    print(
                        f"  step={global_step:4d}/{steps_per_domain} | "
                        f"loss={loss.item():.4f} | "
                        f"bhaya={affective_signals['bhaya_firing_rate']:.3f} | "
                        f"buddhi={affective_signals['buddhi_score']:.3f} | "
                        f"prana={affective_signals['prana_level']:.3f} | "
                        f"vairagya={affective_signals['vairagya_protection_frac']:.3f} | "
                        f"ETA {eta_s/60:.1f}min"
                    )

        train_time = time.time() - domain_start
        print(f"  [Training] Done in {train_time/60:.1f} min")

        # ── Perplexity eval on all domains seen so far ──────────────────────
        print(f"  [Eval] Computing perplexity on {domain_idx + 1} domain(s)...")
        eval_start = time.time()
        for eval_idx in range(domain_idx + 1):
            ppl = evaluate_domain_perplexity(
                model, _eval_loaders[eval_idx], device, EVAL_MAX_BATCHES,
            )
            metrics.record(domain_idx, eval_idx, ppl)
            print(f"    D{eval_idx} ({DOMAINS[eval_idx]}): ppl={ppl:.2f}")
        print(f"  [Eval] Done in {(time.time()-eval_start):.1f}s")

        # ── Domain boundary ──────────────────────────────────────────────────
        boundary_state = affective.on_domain_boundary()
        summary        = metrics.summary()
        logger.log_boundary(boundary_state, summary)
        logger.flush()

        print(f"  [Boundary] Karma pruned: {boundary_state['karma_pruned_weights']:,} weights")
        print(f"  [Boundary] Prana:        {boundary_state['prana_post_boundary']:.3f}")
        print(f"  [Metrics]  AA={summary['AA']:.2f} | BWT={summary['BWT']:.2f} | FWT={summary['FWT']:.2f}")

        domain_total = time.time() - domain_start
        remaining    = len(DOMAINS) - domain_idx - 1
        print(f"  [Timing]   Domain took {domain_total/60:.1f} min | "
              f"~{remaining * domain_total / 60:.0f} min remaining")

    # ── Final results ─────────────────────────────────────────────────────────
    metrics.print_matrix()
    final     = metrics.summary()
    total_min = (time.time() - run_start) / 60
    print(f"\n[Maya-LLM] FINAL — AA={final['AA']:.2f} | BWT={final['BWT']:.2f} | FWT={final['FWT']:.2f}")
    print(f"[Maya-LLM] Total runtime: {total_min:.1f} min")
    print(f"[Maya-LLM] Bhaya Quiescence: {affective.bhaya.is_quiescent()}")
    print(f"[Maya-LLM] Canary: {CANARY_STRING}")

    logger.close()


if __name__ == "__main__":
    main()
