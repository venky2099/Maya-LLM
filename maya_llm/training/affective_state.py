"""
MayaAffectiveState — Unified Affective State Tracker for Maya-LLM v2
Nexus Learning Labs — Maya Research Series

v2 change: Vairagya gradient masking applied during backprop (step-level protection).
High-salience adapter weights have gradients scaled down proportional to protection score.
This is the LLM analogue of EWC — but governed by Vedantic salience, not Fisher information.

Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha
"""

import torch
from maya_llm.plasticity import (
    BhayaMetaplasticity,
    VairagyaDecay,
    BuddhiGate,
    KarmaHistory,
    PranaBudget,
)
from maya_llm.utils.config import CANARY_STRING, VAIRAGYA_GATE_STRENGTH, MAYA_ACTIVE


class MayaAffectiveState:
    def __init__(self, lora_params: list, steps_per_domain: int):
        print(f"[Maya-LLM] Canary: {CANARY_STRING}")
        self.lora_params = lora_params
        self.bhaya    = BhayaMetaplasticity(lora_params)
        self.vairagya = VairagyaDecay(lora_params)
        self.buddhi   = BuddhiGate(steps_per_domain)
        self.karma    = KarmaHistory(lora_params)
        self.prana    = PranaBudget()
        self._prev_loss  = None
        self._domain_idx = 0
        self._step       = 0

    def step(self, loss: float, optimizer) -> dict:
        """
        Called once per training step, after loss.backward(), before optimizer.step().

        v2: Vairagya gradient masking applied here — protected weights resist overwriting.
        Returns dict of affective signals for logging.
        """
        self._step += 1
        loss_delta = (loss - self._prev_loss) if self._prev_loss is not None else 0.0
        self._prev_loss = loss

        # Buddhi advances with experience
        self.buddhi.step()

        # Vairagya accumulates salience from gradient signal
        for p in self.lora_params:
            self.vairagya.accumulate(p, loss_delta)

        # Vairagya gradient masking — core v2 innovation (only when MAYA_ACTIVE)
        # Apply protection mask to gradients before optimizer step.
        # High-salience weights (earned through prior domain training) resist
        # being overwritten by new domain gradients.
        # This is step-level continual learning protection.
        if MAYA_ACTIVE and self.buddhi.is_open():  # only when Maya active
            for p in self.lora_params:
                if p.grad is not None:
                    protection = self.vairagya.get_protection_mask(p)
                    if protection.any():
                        # Scale down gradients on protected weights
                        # protection=1.0 -> gradient scaled by (1 - gate_strength)
                        # protection=0.0 -> gradient unchanged
                        p.grad.data.mul_(1.0 - protection * VAIRAGYA_GATE_STRENGTH)

        # Bhaya fires on loss spike — elevates lability on active params
        vairagya_mask = {
            id(p): self.vairagya.scores[id(p)].mean().item()
            for p in self.lora_params
        }
        bhaya_fired = self.bhaya.update(loss, vairagya_mask) if MAYA_ACTIVE else False

        if bhaya_fired:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    lability = self.bhaya.get_lability(p)
                    group["lr_scale"] = lability

        # Karma accumulates weight trajectory history
        self.karma.accumulate()
        self.karma.decay()

        # Prana gates effective LR metabolically
        grad_mag      = self._mean_grad_magnitude()
        mean_act      = min(1.0, grad_mag)
        vairagya_mean = self.vairagya.protection_fraction()
        prana_level   = self.prana.update(grad_mag, mean_act, vairagya_mean)

        return {
            "bhaya_fired":              bhaya_fired,
            "bhaya_firing_rate":        self.bhaya.firing_rate,
            "buddhi_score":             self.buddhi.score,
            "vairagya_protection_frac": vairagya_mean,
            "karma_pruned_frac":        self.karma.pruned_fraction,
            "prana_level":              prana_level,
        }

    def on_domain_boundary(self) -> dict:
        """
        Called at end of each domain.
        Applies Vairagya boundary decay, Karma pruning, Prana recovery.
        Resets Buddhi for next domain.
        """
        self._domain_idx += 1

        self.vairagya.snapshot_domain()  # snapshot before decay — domain-selective
        self.vairagya.apply_boundary_decay()
        self.vairagya.decay_scores()

        pruned = self.karma.prune_at_boundary(self.buddhi.score) if MAYA_ACTIVE else 0

        self.prana.boundary_recovery()
        self.buddhi.reset_for_domain()

        return {
            "domain":                   self._domain_idx,
            "karma_pruned_weights":     pruned,
            "prana_post_boundary":      self.prana.p,
            "vairagya_protection_frac": self.vairagya.protection_fraction(),
            "buddhi_score_at_boundary": self.buddhi.score,
        }

    def _mean_grad_magnitude(self) -> float:
        total, count = 0.0, 0
        for p in self.lora_params:
            if p.grad is not None:
                total += p.grad.data.abs().mean().item()
                count += 1
        return total / max(1, count)

    def full_state(self) -> dict:
        return {
            "bhaya":    self.bhaya.state_dict(),
            "vairagya": self.vairagya.state_dict(),
            "buddhi":   self.buddhi.state_dict(),
            "karma":    self.karma.state_dict(),
            "prana":    self.prana.state_dict(),
        }
