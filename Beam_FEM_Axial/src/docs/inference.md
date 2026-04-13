# `inference.py` — Inference & Evaluation Script

## Purpose

Loads the trained BeamNet model and runs it on test cases.  
Computes error metrics, saves comparison plots as PNG images, and optionally exports `.vtu` files that can be visualised in **ParaView**.

---

## Usage

```bash
# Evaluate all test cases (metrics + PNG)
python3 src/inference.py

# Also write .vtu files for ParaView visualisation
python3 src/inference.py --vtu

# Evaluate a single case only
python3 src/inference.py --case case_0100

# Single case with VTU output
python3 src/inference.py --case case_0100 --vtu
```

---

## Outputs

| Output | Description |
|--------|-------------|
| `saves/prediction_<case_id>.png` | 3-panel comparison plot |
| `saves/<case_id>_comparison.vtu` | ParaView file (only with `--vtu`) |
| stdout | MAE, RMSE, relative error per case |

---

## Helper functions

### `load_model()`

Loads `saves/beam_pinn.pt`, reconstructs the `BeamNet` with the saved hidden size, and sets it to evaluation mode (`model.eval()`).

### `load_norm_params()`

Loads `saves/norm_params.npz` — the normalisation statistics saved during training.

### `build_features(coords, params)`

Assembles the `(N, 9)` input matrix for one case from node coordinates and case parameters — same layout as during training.

### `normalise_X(X, p)`

Applies log-transform to the force column and z-scores all features using the saved statistics.

### `predict(model, X_n, p)`

Runs the model in no-gradient mode (`torch.no_grad()`), then **denormalises** the output:

```
U_mm = U_normalised × Y_std + Y_mean
```

Returns displacements in mm.

---

## Metrics (`compute_metrics`)

Three error metrics are computed for each test case:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | `mean(|pred − true|)` | Average absolute error per component [mm] |
| **RMSE** | `sqrt(mean((pred − true)²))` | Root mean squared error [mm] |
| **Relative error** | `mean(‖pred − true‖) / mean(‖true‖) × 100` | Percentage error on displacement magnitude |

---

## PNG plot (`save_png`)

Produces a 3-panel 3D scatter plot of the beam nodes, coloured by displacement magnitude `|U|`:

| Panel | Content |
|-------|---------|
| Left | FEM ground truth |
| Centre | BeamNet prediction |
| Right | Absolute error `|Δ|` (red colormap) |

The left and centre panels share the same colour scale for direct visual comparison.

---

## VTU output (`save_vtu`)

Creates a new `.vtu` mesh file (same geometry as the FEM result) with three extra point data fields:

| Field | Content |
|-------|---------|
| `displacement_FEM` | Ground truth from simulation |
| `displacement_pred` | BeamNet prediction |
| `error_magnitude` | Signed difference of displacement magnitudes |

Open in ParaView and colour by any of these fields to inspect where the model is most/least accurate.

---

## Main flow

```
1. Parse --case and --vtu arguments
2. Load model + normalisation params
3. Load vtk_manifest.json to find test cases
4. For each target case:
   a. Read the VTU file (coordinates + FEM displacements)
   b. Build and normalise features
   c. Run BeamNet → get predicted displacements
   d. Compute MAE / RMSE / relative error
   e. Save PNG comparison plot
   f. (optional) Save VTU file
5. Print ParaView instructions if VTU files were written
```
