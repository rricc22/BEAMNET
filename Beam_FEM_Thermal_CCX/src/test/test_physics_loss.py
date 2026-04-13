"""
test_physics_loss.py — unit tests for physics_loss in losses.py

Tests:
  1. Quadratic model  → ∇²T̂ = 6 ≠ 0     → loss > 0
  2. k-weighting      → high-k samples contribute more than low-k
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
from src.losses import physics_loss


class QuadraticModel(nn.Module):
    """T = x² + y² + z²  →  ∇²T = 6 ≠ 0"""
    def forward(self, x):
        return (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2).unsqueeze(1)


N = 100
torch.manual_seed(0)
x_norm = torch.randn(N, 6)


def test_quadratic_model_nonzero_loss():
    loss = physics_loss(QuadraticModel(), x_norm)
    print(f"Quadratic model loss (expect > 0): {loss.item():.4f}")
    assert loss.item() > 0.1, f"FAIL: got {loss.item()}"


def test_k_weighting():
    x_low_k = x_norm.clone();  x_low_k[:, 4]  = 0.5
    x_high_k = x_norm.clone(); x_high_k[:, 4] = 2.0

    loss_low  = physics_loss(QuadraticModel(), x_low_k)
    loss_high = physics_loss(QuadraticModel(), x_high_k)
    print(f"Low-k loss:  {loss_low.item():.4f}")
    print(f"High-k loss: {loss_high.item():.4f}")
    assert loss_high.item() > loss_low.item(), (
        f"FAIL: high-k ({loss_high.item()}) should exceed low-k ({loss_low.item()})"
    )


if __name__ == "__main__":
    test_quadratic_model_nonzero_loss()
    test_k_weighting()
    print("\nAll tests passed.")
