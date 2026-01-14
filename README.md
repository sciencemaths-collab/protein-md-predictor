# Protein Molecular Dynamics Predictor (v1)

**Release:** v1.0.3 – fixes 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencemaths-collab/protein-md-predictor/blob/main/notebooks/Run_on_Google_Colab.ipynb)

> **Note:** Colab demo links are temporary. The public URL is alive only while the Colab runtime is running.

**Protein MD Predictor** is a Streamlit app for **molecular dynamics / structure prediction** using a **physics-guided tokenized dynamics** pipeline.

This is the **first public rollout (v1)**. Expect frequent updates, refinements, and new capabilities. Use it, stress-test it, and have fun with it.

## What it does
- Upload a topology + trajectory (or use a synthetic demo).
- Extract invariant motion features (distances / signals).
- Tokenize the trajectory (PCA + vector quantization codebook).
- Train a forecasting engine with physics-shaped losses.
- Predict forward to a user-defined horizon.
- Decode predictions under geometric / physics constraints.
- Export a **multi-model PDB** for “future movie” playback in PyMOL / VMD / ChimeraX.

## Quick start (local)

### Requirements
- Python **3.11+** (tested with 3.11.8)

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

Then open the URL shown in your terminal.

## Quick start (Google Colab)
- Open the notebook: `notebooks/Run_on_Google_Colab.ipynb`
- Run all cells
- Colab will print a **public URL** you can click to open the Streamlit app.

**Tip:** Colab is best for demos and smaller uploads. For heavy workloads or very large trajectories, run locally or deploy on a dedicated server.

## Deploy options

### Streamlit Community Cloud
1. Push this repo to GitHub
2. Streamlit Cloud → “New app”
3. Select repo + branch
4. Main file: `app.py`

**Note:** Streamlit Cloud has tighter upload limits than a local run. For large inputs, use local/server deployment.

### Hugging Face Spaces
Create a Space with **Streamlit** SDK and point it to this repo. Spaces are great for discovery and sharing.

## Repo hygiene (important)
Do **not** commit large trajectories to GitHub. This repo’s `.gitignore` excludes common MD binaries (`*.xtc`, `*.trr`, `*.dcd`, etc.).

## Contact
- Email: **bessuman.academia@gmail.com**
- Website: **www.proteinsimulationconsulting.com**
