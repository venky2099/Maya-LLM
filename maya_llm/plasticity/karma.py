"""
Karma â€” Second-Order Plasticity History on LoRA Adapter Weights
Nexus Learning Labs â€” Maya Research Series

Biological grounding: Microglial synaptic pruning via C1q/C3 complement cascade.
Karma = absolute integral of per-weight trajectory changes across domains.
High-Karma weights = chronic cross-domain interference â†’ pruning candidates at boundary.
ORCID magic number 0.002315 embedded in decay rate (series convention P6+).
"""

import torch
from maya_llm.utils.config import (
    KARMA_DECAY_RATE,
    KARMA_THRESHOLD,
    KARMA_ACCUMULATION_MODE,
)


class KarmaHistory:
    def __init__(self, lora_params: list):
        self.lora_params = lora_params
        self.scores = {id(p): torch.zeros_like(p.data) for p in lora_params}
        self.masks = {id(p): torch.ones_like(p.data) for p in lora_params}
        self._prev_weights = {id(p): p.data.clone() for p in lora_params}
        self._pruned_fraction = 0.0

    def accumulate(self) -> None:
        """Per-step: accumulate absolute weight trajectory changes."""
        for p in self.lora_params:
            pid = id(p)
            if self.masks[pid].sum() == 0:
                continue
            delta = (p.data - self._prev_weights[pid]).abs()
            self.scores[pid] += delta * self.masks[pid]
            self._prev_weights[pid] = p.data.clone()

    def decay(self) -> None:
        """Inter-step slow decay â€” ORCID magic number rate."""
        for pid in self.scores:
            self.scores[pid] *= (1.0 - KARMA_DECAY_RATE)

    def prune_at_boundary(self, buddhi_score: float) -> int:
        """
        At domain boundary: zero-mask high-Karma weights.
        Buddhi modulates threshold â€” young model prunes aggressively,
        mature model prunes conservatively. Mirrors P8 developmental arc.
        Returns count of pruned weights.
        """
        effective_threshold = KARMA_THRESHOLD * (0.5 + buddhi_score * 0.5)
        total_pruned = 0
        for p in self.lora_params:
            pid = id(p)
            prune_candidates = (self.scores[pid] > effective_threshold) & (self.masks[pid] == 1.0)
            count = prune_candidates.sum().item()
            if count > 0:
                with torch.no_grad():
                    p.data[prune_candidates] = 0.0
                    self.masks[pid][prune_candidates] = 0.0
            total_pruned += int(count)
        self._pruned_fraction = self._compute_pruned_fraction()
        return total_pruned

    def _compute_pruned_fraction(self) -> float:
        total, pruned = 0, 0
        for pid, mask in self.masks.items():
            pruned += (mask == 0.0).sum().item()
            total += mask.numel()
        return pruned / max(1, total)

    @property
    def pruned_fraction(self) -> float:
        return self._pruned_fraction

    def state_dict(self) -> dict:
        return {
            "scores": {pid: s.cpu() for pid, s in self.scores.items()},
            "pruned_fraction": self._pruned_fraction,
        }


