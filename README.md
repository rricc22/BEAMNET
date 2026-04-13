# BEAMNET

Physics-Informed Neural Networks as FEM surrogates for beam structures.

- **ThermalNet** — steady-state heat conduction (Elmer FEM, 500 cases)
- **BeamNet** — 3D linear elasticity (CalculiX FEM, 1 575 cases)

Full report: `Riccardo_castellano_LabProject.pdf`

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/rricc22/BEAMNET
cd BEAMNET
```

### 2. Install dependencies

```bash
conda create -n beamnet python=3.11 -y
conda activate beamnet
pip install -r Beam_FEM_Thermal_CCX/requirements.txt
pip install -r Beam_FEM_Axial/requirements.txt
```

### 3. Pull the datasets (DVC)

Configure the remote with the credentials provided separately (see report title page):

```bash
dvc remote add -d storage webdavs://nextcloud.rricc22-homelab.com/remote.php/dav/files/admin/dvc-storage
dvc remote modify storage user <user>
dvc remote modify storage password <password>
dvc pull
```

This downloads:
- `Beam_FEM_Thermal_CCX/elmer_cases/` — 3.3 GB
- `Beam_FEM_Axial/ccx_cases/` — 20 GB

---

## Run inference

### ThermalNet

```bash
cd Beam_FEM_Thermal_CCX
python3 src/inference.py
```

### BeamNet

```bash
cd Beam_FEM_Axial
python3 src/inference.py
```

---

## Project structure

```
BEAMNET/
├── Beam_FEM_Thermal_CCX/
│   ├── src/          # train.py, inference.py, arch.py, losses.py
│   ├── utils/        # visualize_results.py, lambda_study.py
│   └── scripts/      # FEM dataset generation pipeline
├── Beam_FEM_Axial/
│   ├── src/          # train.py, inference.py, arch.py, losses.py
│   ├── utils/        # visualize_results.py
│   └── scripts/      # FEM dataset generation pipeline
└── Riccardo_castellano_LabProject.pdf
```
