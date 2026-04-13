"""
BeamNet: predicts nodal displacement (U1, U2, U3) in mm.

Input  (9 features):
  x, y, z      – node coordinates in the undeformed mesh      [mm]   (3)
  fx, fy, fz   – force direction unit vector                  [-]    (3)
  F            – total applied force magnitude                [N]    (1)
  E, nu        – material: Young's modulus, Poisson's ratio   [MPa,-](2)

Output (3):
  U1, U2, U3  – nodal displacement components                [mm]
"""

import torch
import torch.nn as nn


class BeamNet(nn.Module):
    """Simple 4-hidden-layer MLP for beam structural response prediction."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
