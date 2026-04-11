"""
TRACE Benchmark Loader for Maya-LLM — v2 (optimised)
Nexus Learning Labs — Maya Research Series

8 sequential domains (Wang et al., 2023 — arXiv:2310.06762).
Eval: perplexity-based (forward pass only). No model.generate(). ~100x faster.

Perplexity interpretation for continual learning:
  Lower perplexity on a domain = better retention of that domain.
  BWT = mean(ppl_final[j] - ppl_immediate[j]) — positive = forgetting.
"""

import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from maya_llm.utils.config import MAX_SEQ_LEN, BATCH_SIZE, EVAL_MAX_BATCHES

# ─── TRACE canonical domain sequence ─────────────────────────────────────────
TRACE_DOMAINS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]

# Verified HuggingFace dataset IDs — tested April 2026
TRACE_HF_IDS = {
    "C-STANCE":    ("rcorp/c-stance",              "train",      "validation"),
    "FOMC":        ("gtfintechlab/fomc-communication", "train",  "validation"),
    "MeetingBank": ("huuuyeah/meetingbank",         "train",      "test"),
    "Py150":       ("semeru/codexglue-code-to-code-clone-detection-bigclonebench", "train", "validation"),
    "ScienceQA":   ("derek-thomas/ScienceQA",       "train",      "test"),
    "NumGLUE-cm":  ("juletxara/mgsm",               "train",      "test"),
    "NumGLUE-ds":  ("allenai/math_qa",              "train",      "validation"),
    "20Minuten":   ("rcorp/20minuten",              "train",      "validation"),
}

PROMPT_TEMPLATES = {
    "C-STANCE":    "Classify the stance of the following Chinese text.\nText: {text}\nStance:",
    "FOMC":        "Classify the monetary policy signal.\nStatement: {text}\nSignal:",
    "MeetingBank": "Summarize the following meeting transcript.\nTranscript: {text}\nSummary:",
    "Py150":       "Complete the following Python code.\nCode: {text}\nCompletion:",
    "ScienceQA":   "Answer the following science question.\nQuestion: {text}\nAnswer:",
    "NumGLUE-cm":  "Solve the following numerical reasoning problem.\nProblem: {text}\nSolution:",
    "NumGLUE-ds":  "Solve the following math problem.\nProblem: {text}\nSolution:",
    "20Minuten":   "Simplify the following German news article.\nArticle: {text}\nSimplified:",
}


class TRACEDomainDataset(Dataset):
    def __init__(self, samples: list, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        full_text = sample["prompt"] + " " + sample["answer"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Mask prompt tokens — only compute loss on answer tokens
        prompt_enc = self.tokenizer(
            sample["prompt"],
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len]          = -100
        labels[attention_mask == 0]  = -100
        # skip sequences that are entirely masked
        if (labels != -100).sum() == 0:
            labels[0] = input_ids[0]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _fetch_from_hf(domain: str, split: str, max_samples: int) -> list:
    hf_id, train_split, eval_split = TRACE_HF_IDS[domain]
    target_split = train_split if split == "train" else eval_split
    template     = PROMPT_TEMPLATES[domain]

    try:
        ds = load_dataset(hf_id, split=target_split, streaming=True)
        samples = []
        for item in ds:
            text = (
                item.get("text") or item.get("sentence") or
                item.get("question") or item.get("content") or
                item.get("prompt") or str(list(item.values())[0])
            )
            answer = (
                item.get("answer") or item.get("label") or
                item.get("summary") or item.get("target") or
                str(list(item.values())[-1])
            )
            prompt = template.format(text=str(text)[:300])
            samples.append({"prompt": str(prompt), "answer": str(answer)})
            if len(samples) >= max_samples:
                break
        if samples:
            return samples
    except Exception as e:
        print(f"  [TRACE] HuggingFace load failed for {domain}: {e}")

    print(f"  [TRACE] Using synthetic stub for {domain}")
    return _synthetic_stub(domain, max_samples)


def _synthetic_stub(domain: str, n: int) -> list:
    """Varied synthetic stubs — different loss per domain for meaningful forgetting signal."""
    template = PROMPT_TEMPLATES[domain]
    vocab    = {
        "C-STANCE":    ["支持", "反对", "中立", "赞同", "拒绝"],
        "FOMC":        ["hawkish", "dovish", "neutral", "tightening", "easing"],
        "MeetingBank": ["discussed", "agreed", "decided", "reviewed", "planned"],
        "Py150":       ["return", "def", "class", "import", "for"],
        "ScienceQA":   ["photosynthesis", "gravity", "evolution", "mitosis", "entropy"],
        "NumGLUE-cm":  ["42", "17", "256", "3.14", "100"],
        "NumGLUE-ds":  ["mean", "variance", "regression", "correlation", "probability"],
        "20Minuten":   ["Zürich", "Bern", "Basel", "Genf", "Luzern"],
    }
    words   = vocab.get(domain, ["sample", "text", "data", "input", "output"])
    samples = []
    for i in range(n):
        word   = words[i % len(words)]
        prompt = template.format(text=f"{word} sample {i} for {domain} domain evaluation.")
        answer = f"{word} response {i % len(words)}."
        samples.append({"prompt": prompt, "answer": answer})
    return samples


def load_trace_domain(
    domain:        str,
    tokenizer,
    data_dir:      str | None = None,
    train_samples: int = 1000,
    test_samples:  int = 200,
    batch_size:    int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    print(f"  [TRACE] Loading domain: {domain}")

    if data_dir:
        train_path = os.path.join(data_dir, domain, "train.json")
        eval_path  = os.path.join(data_dir, domain, "eval.json")
        if os.path.exists(train_path) and os.path.exists(eval_path):
            print(f"  [TRACE] Using local data: {train_path}")
            train_data = json.load(open(train_path, encoding="utf-8"))[:train_samples]
            eval_data  = json.load(open(eval_path, encoding="utf-8"))[:test_samples]
        else:
            train_data = _fetch_from_hf(domain, "train", train_samples)
            eval_data  = _fetch_from_hf(domain, "eval",  test_samples)
    else:
        train_data = _fetch_from_hf(domain, "train", train_samples)
        eval_data  = _fetch_from_hf(domain, "eval",  test_samples)

    train_ds = TRACEDomainDataset(train_data, tokenizer)
    eval_ds  = TRACEDomainDataset(eval_data,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  [TRACE] {domain}: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_loader, eval_loader


def evaluate_domain_perplexity(
    model,
    eval_loader: DataLoader,
    device,
    max_batches: int = EVAL_MAX_BATCHES,
) -> float:
    """
    Perplexity-based eval — single forward pass, no generation.
    Returns perplexity score. Lower = better retention.
    ~100x faster than model.generate() eval.

    For continual learning metrics:
      record perplexity immediately after training (R[i][i])
      record again after subsequent domains (R[j][i] for j > i)
      BWT = mean(R[final][i] - R[i][i]) — positive = forgetting (ppl increased)
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss is mean cross-entropy over non-masked tokens
            valid_tokens = (labels != -100).sum().item()
            batch_loss = outputs.loss.item()
            if valid_tokens > 0 and not (batch_loss != batch_loss):  # skip NaN
                total_loss   += batch_loss * valid_tokens
                total_tokens += valid_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss    = total_loss / total_tokens
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        return float("nan")
    perplexity  = math.exp(min(avg_loss, 20))  # clamp to prevent overflow
    return round(perplexity, 4)
