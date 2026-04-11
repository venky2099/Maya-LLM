"""
Vairagya â€” Heterosynaptic Wisdom-Governed Decay on LoRA Adapter Weights
Nexus Learning Labs â€” Maya Research Series

Biological grounding: Heterosynaptic BCM sliding threshold (Bienenstock et al., 1982).
High-salience adapter weights accumulate Vairagya protection scores.
Low-salience adapter weights decay toward zero between domains.
Salience = accumulated task performance contribution, not instantaneous gradient.
"""

import torch
from maya_llm.utils.config import (
    VAIRAGYA_TAU,
    VAIRAGYA_DECAY_RATE,
    VAIRAGYA_PROTECTION_THRESH,
    VAIRAGYA_GATE_STRENGTH,
)


class VairagyaDecay:
    def __init__(self, lora_params: list):
        self.lora_params = lora_params
        # Per-parameter salience scores â€” accumulated over training
        self.scores = {id(p): torch.zeros_like(p.data) for p in lora_params}
        self._param_map = {id(p): p for p in lora_params}
        self._domain_snapshots = {}
        self._current_domain = 0
        self._domain_snapshots = {}
        self._current_domain = 0
        self._domain_snapshots = {}
        self._current_domain = 0

    def accumulate(self, param: torch.nn.Parameter, loss_delta: float) -> None:
        """Update salience score for a parameter based on its loss contribution."""
        pid = id(param)
        if pid not in self.scores:
            return
        if param.grad is not None:
            # Salience = gradient magnitude weighted by loss reduction
            contribution = param.grad.data.abs() * max(0.0, -loss_delta)
            self.scores[pid] = self.scores[pid] * (1.0 - 1.0 / VAIRAGYA_TAU) + contribution

    def snapshot_domain(self) -> None:
        """
        Snapshot peak salience for current domain before boundary decay.
        Called at domain boundary BEFORE decay_scores().
        Stores which weights were most important FOR THIS DOMAIN SPECIFICALLY.
        """
        snapshot = {}
        for p in self.lora_params:
            pid = id(p)
            snapshot[pid] = self.scores[pid].clone()
        self._domain_snapshots[self._current_domain] = snapshot
        self._current_domain += 1

    def apply_boundary_decay(self) -> None:
        """
        At each domain boundary: decay low-salience adapter weights.
        High-salience weights (score > threshold) are protected.
        Mirrors BCM-boundary decay from Maya-CL (P3).
        """
        for p in self.lora_params:
            pid = id(p)
            score = self.scores[pid]
            norm_score = score / (score.max().clamp(min=1e-8))
            protection_mask = (norm_score >= VAIRAGYA_PROTECTION_THRESH).float()
            decay_mask = 1.0 - protection_mask * VAIRAGYA_GATE_STRENGTH
            with torch.no_grad():
                p.data.mul_(decay_mask)

    def decay_scores(self) -> None:
        """Slow inter-domain score decay â€” ORCID magic number rate."""
        for pid in self.scores:
            self.scores[pid] *= (1.0 - VAIRAGYA_DECAY_RATE)

    def get_protection_mask(self, param: torch.nn.Parameter) -> torch.Tensor:
        """
        Domain-selective top-K% protection.
        Protection is built from per-domain salience snapshots, not cumulative scores.
        Each prior domain contributes a recency-weighted protection signal.
        Domain i-1 (most recent) contributes most; older domains contribute less.
        This prevents protecting domain-general weights — only domain-specific peaks protected.
        """
        pid = id(param)
        if not self._domain_snapshots:
            # No completed domains yet — no protection during first domain
            return torch.zeros(param.data.shape, device=param.data.device)

        device = param.data.device
        combined = torch.zeros_like(param.data)
        n_snapshots = len(self._domain_snapshots)

        for domain_idx, snapshot in self._domain_snapshots.items():
            if pid not in snapshot:
                continue
            domain_score = snapshot[pid].to(device)
            flat = domain_score.flatten()
            if flat.max() < 1e-10:
                continue
            # Top 10% for this specific domain
            k = max(1, int(flat.numel() * 0.10))
            threshold = torch.topk(flat, k).values[-1]
            domain_mask = (domain_score >= threshold).float()
            # Recency weighting: most recent domain gets weight 1.0,
            # older domains decay by 0.5 per step back
            recency = 0.5 ** (n_snapshots - 1 - domain_idx)
            combined = torch.maximum(combined, domain_mask * recency)

        # Binary threshold: protect if combined signal >= 0.5
        # This means only weights protected by at least the most recent domain
        return (combined >= 0.5).float()

    def protection_fraction(self) -> float:
        total, protected = 0, 0
        for p in self.lora_params:
            pid = id(p)
            score = self.scores[pid]
            flat = score.flatten()
            if flat.max() >= 1e-10:
                k = max(1, int(flat.numel() * 0.10))
                threshold = torch.topk(flat, k).values[-1]
                protected += (score >= threshold).sum().item()
            total += score.numel()
        return protected / max(1, total)

    def state_dict(self) -> dict:
        return {
            "scores": {pid: s.cpu() for pid, s in self.scores.items()},
            "protection_fraction": self.protection_fraction(),
        }


