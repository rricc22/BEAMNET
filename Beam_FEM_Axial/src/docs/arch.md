# `arch.py` — Neural Network Architecture

## Purpose

Defines **BeamNet**, the neural network that predicts how a beam deforms under a load.  
Given a node's position and the loading/material conditions, it outputs the 3D displacement of that node.

---

## What the network does

**Inputs (9 numbers per node):**

| Feature | Description | Unit |
|---------|-------------|------|
| `x, y, z` | Node position in the undeformed beam | mm |
| `fx, fy, fz` | Unit vector describing the force direction | — |
| `F` | Total applied force magnitude | N |
| `E` | Young's modulus (stiffness of the material) | MPa |
| `nu` | Poisson's ratio (how much the material squishes sideways) | — |

**Outputs (3 numbers per node):**

| Output | Description | Unit |
|--------|-------------|------|
| `U1` | Displacement along X | mm |
| `U2` | Displacement along Y | mm |
| `U3` | Displacement along Z | mm |

---

## Architecture

BeamNet is a **Multi-Layer Perceptron (MLP)** — a simple feedforward neural network with 4 hidden layers.

```
Input (9)
  → Linear(9 → 256) + Tanh
  → Linear(256 → 256) + Tanh
  → Linear(256 → 256) + Tanh
  → Linear(256 → 256) + Tanh
  → Linear(256 → 3)
Output (3)
```

- **Hidden size**: 256 neurons per layer (configurable via `hidden` argument)
- **Activation**: `Tanh` — chosen because it is smooth and infinitely differentiable, which matters for computing the physics-based loss (second-order gradients in `losses.py`)
- **Total parameters**: ~330 000 (with `hidden=256`)

---

## Code

```python
class BeamNet(nn.Module):
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
```

The `forward` method simply passes the input through all layers in sequence.

---

## Key design choices

- **No activation on the output layer** — displacement can be any real number (positive or negative), so no squashing function is applied at the end.
- **Tanh instead of ReLU** — ReLU has zero second derivatives almost everywhere, which would make the physics loss in `losses.py` useless. Tanh is smooth everywhere.
- **Inherits from `nn.Module`** — standard PyTorch pattern; allows the model to be trained, saved, and loaded with standard PyTorch utilities.
