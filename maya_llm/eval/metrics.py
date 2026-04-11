"""
Continual Learning Metrics for Maya-LLM — v2 (perplexity-based)
Nexus Learning Labs — Maya Research Series

Perplexity interpretation:
  R[i][j] = perplexity on domain j after training on domain i.
  Lower perplexity = better. Forgetting = perplexity increases over time.

  AA  = mean perplexity across all domains after final training (lower is better)
  BWT = mean(R[n-1][j] - R[j][j]) for j < n  (positive = forgetting)
  FWT = mean(R[i-1][i]) for i > 0             (lower = better forward transfer)

For reporting in paper: AA and BWT are the primary metrics.
BWT > 0 means the model forgot. Condition F should show lower BWT than Condition A.
"""

import numpy as np


class CLMetrics:
    def __init__(self, num_domains: int):
        self.n = num_domains
        self.R = np.full((num_domains, num_domains), np.nan)  # perplexity matrix

    def record(self, trained_on: int, evaluated_on: int, perplexity: float) -> None:
        self.R[trained_on][evaluated_on] = perplexity

    def average_accuracy(self) -> float:
        """Mean perplexity across all domains after final training. Lower = better."""
        final_row = self.R[self.n - 1]
        valid = final_row[~np.isnan(final_row)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def backward_transfer(self) -> float:
        """
        BWT = mean(R[n-1][j] - R[j][j]) for j < n.
        Positive = forgetting (perplexity increased). Higher is worse.
        """
        deltas = []
        for j in range(self.n - 1):
            if not np.isnan(self.R[self.n - 1][j]) and not np.isnan(self.R[j][j]):
                deltas.append(self.R[self.n - 1][j] - self.R[j][j])
        return float(np.mean(deltas)) if deltas else 0.0

    def forward_transfer(self) -> float:
        """FWT = mean perplexity on next domain before training it."""
        deltas = []
        for i in range(1, self.n):
            if not np.isnan(self.R[i - 1][i]):
                deltas.append(self.R[i - 1][i])
        return float(np.mean(deltas)) if deltas else 0.0

    def summary(self) -> dict:
        return {
            "AA":  round(self.average_accuracy(), 2),
            "BWT": round(self.backward_transfer(), 2),
            "FWT": round(self.forward_transfer(), 2),
        }

    def print_matrix(self) -> None:
        print("\nPerplexity Matrix R[trained_on][eval_on] (lower = better retention):")
        header = "         " + "  ".join([f"  D{j}" for j in range(self.n)])
        print(header)
        for i in range(self.n):
            row = f"D{i} →  " + "  ".join(
                [f"{self.R[i][j]:6.1f}" if not np.isnan(self.R[i][j]) else "   ---" for j in range(self.n)]
            )
            print(row)
        print(f"\nAA (final mean ppl):  {self.average_accuracy():.2f}")
        print(f"BWT (forgetting):     {self.backward_transfer():.2f}  (positive = forgot)")
        print(f"FWT (transfer):       {self.forward_transfer():.2f}")
