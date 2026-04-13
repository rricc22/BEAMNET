"""
ThermalNet: predicts nodal temperature T [°C] for steady-state heat conduction.

Input  (6 features):
  x, y, z  – node coordinates in mm                          [mm]  (3)
  q        – total applied heat flux (at Nload face)         [mW]  (1)
  k        – material thermal conductivity                [mW/mm/°C](1)
  T_fix    – fixed temperature at Nfix face (BC Dirichlet)   [°C]  (1)

Output (1):
  T        – nodal temperature                               [°C]
"""

import torch
import torch.nn as nn


class ThermalNet(nn.Module):
    """Simple 4-hidden-layer MLP for beam steady-state thermal response."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, 1)
