# `train.py` — Training Script

## Purpose

Loads FEM simulation data, normalises it, and trains **BeamNet** as a Physics-Informed Neural Network (PINN).  
Saves the trained model weights and normalisation parameters for later use by `inference.py`.

---

## Data sources

| File | Content |
|------|---------|
| `ccx_cases/elasticity_axial_beam/*/job.vtu` | Node coordinates + FEM displacement results |
| `ccx_cases/elasticity_axial_beam/*/case_params.json` | Load direction, force magnitude, material (E, nu) |
| `ccx_cases/elasticity_axial_beam/vtk_manifest.json` | Index of all cases and their train/test split |

### Load directions (`DIR_MAP`)

Maps a string key to a 3D unit vector:

| Key | Direction | Meaning |
|-----|-----------|---------|
| `Y` | `[0, 1, 0]` | Transverse bending in Y |
| `Z` | `[0, 0, 1]` | Transverse bending in Z (downward) |
| `YZ` | `[0, 0.707, 0.707]` | Biaxial diagonal |
| `X+` | `[1, 0, 0]` | Axial tension |
| `X-` | `[-1, 0, 0]` | Axial compression |

---

## Hyperparameters (`CONFIG`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `hidden` | 256 | Neurons per hidden layer |
| `epochs` | 100 | Training iterations over the full dataset |
| `batch_size` | 4096 | Nodes per data batch |
| `phys_batch_size` | 1024 | Nodes used for physics loss per step |
| `lr` | 1e-3 | Initial learning rate |
| `lambda_physics` | 0.1 | Weight of the physics loss term |
| `step_size` | 30 | Epochs between learning rate drops |
| `gamma` | 0.5 | LR multiplied by this at each step (halved every 30 epochs) |

---

## Data pipeline

### 1. `load_dataset()`

Reads every successful VTU case from the manifest and assembles one large array:

- **X** shape `(N_total, 9)`: `[x, y, z, fx, fy, fz, F, E, nu]`
- **Y** shape `(N_total, 3)`: `[U1, U2, U3]` displacements in mm
- **splits**: `'train'` or `'test'` label per row

### 2. Normalisation

Raw features span very different scales (mm vs. N vs. MPa), so everything is **z-scored**:

```
x̂ = (x − mean) / std
```

Special treatment for **force magnitude** (column 6): it spans several orders of magnitude, so a **log-transform** is applied first before z-scoring.

- `compute_norm_params(X_tr, Y_tr)` — computes mean and std from training data only (avoids data leakage)
- `apply_norm(X, Y, norm)` — applies the transform to any split

Normalisation statistics are saved to `saves/norm_params.npz` for use at inference time.

---

## Training loop

### Total loss

```
loss = data_MSE + λ_physics × physics_loss
```

- **data_MSE**: standard mean squared error between predicted and FEM displacements
- **physics_loss**: Navier-Cauchy residual (from `losses.py`) computed on a subset of the batch

### Per-epoch steps

1. **Forward pass** through `BeamNet` on a batch of nodes
2. Compute `data_MSE`
3. Compute `physics_loss` on the first `phys_batch_size` nodes
4. Backpropagate the combined loss
5. **Gradient clipping** (`max_norm=1.0`) to stabilise training
6. `Adam` optimiser step
7. After each epoch: evaluate on the full validation set, compute MAE in mm
8. **Save the best model** (lowest validation MSE) to `saves/beam_pinn.pt`
9. Step the learning rate scheduler (halve LR every 30 epochs)

### Logging

If `wandb` is installed, all metrics are logged to a Weights & Biases project called `beam-pinn`.

---

## Outputs

| File | Content |
|------|---------|
| `saves/beam_pinn.pt` | Best model checkpoint (weights + config) |
| `saves/norm_params.npz` | Normalisation statistics (mean, std for X and Y) |

---

## How to run

```bash
python3 src/train.py
```

Progress is printed every 10 epochs:

```
 Epoch    Data MSE   Phys loss     Val MSE   Val MAE[mm]
----------------------------------------------------------
     1    0.123456    0.004321    0.098765      0.012345
    10    0.045678    0.001234    0.034567      0.007890
   ...
```
