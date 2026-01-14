# Protein MD Predictor (v1)

**Protein MD Predictor** is a Streamlit app for **molecular dynamics and structure prediction** using a **physics-guided tokenized dynamics** pipeline.

This is the **first public rollout (v1)**. Expect frequent updates, improvements, and new capabilities. Use it, stress-test it, and have fun with it.

## What it does
- Upload a topology + trajectory (or use a synthetic demo).
- Build a compact token representation of motion.
- Train a forecasting engine.
- Predict forward to a user-defined horizon.
- Decode predictions under geometric/physics constraints.
- Export a **multi-model PDB** for "future movie" playback in PyMOL/VMD/Chimera.

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

Open the local URL Streamlit prints in your terminal.

## How to use (straight to the point)
1. In the sidebar, choose **Upload** (or **URL** if you host files elsewhere).
2. Provide **Topology** (e.g., PDB/GRO/PSF) and **Trajectory** (e.g., XTC/DCD).
3. Click **Build + Train engine**.
4. Set prediction horizon and click **Predict**.
5. Optional: open **Future movie export** and generate a multi-model PDB.

Notes:
- For large files, the app streams uploads to disk to avoid memory spikes.
- Hosting providers may still impose their own upload/runtime limits.

## Quick start (Google Colab)

You can run the app in a hosted notebook session (no local install) and get a temporary public URL.

- Open: `notebooks/Run_on_Google_Colab.ipynb`
- Click **Runtime → Run all**
- The notebook prints a **Public URL** (trycloudflare). Open it to use the app.

**Tip:** Colab is best for demos and smaller uploads. For heavy workloads, run locally or deploy on a dedicated server.

## Deploy (GitHub → Streamlit Cloud)
1. Create a GitHub repo and push this project.
2. On Streamlit Community Cloud, create a new app and set:
   - **Main file**: `app.py`
   - **Python**: 3.11+
   - **Requirements**: `requirements.txt`

## Deploy (Hugging Face Spaces)
1. Create a new Space (SDK: **Streamlit**).
2. Upload/push repo contents.
3. Ensure `app.py` and `requirements.txt` are in the Space root.

## Contact
- Email: **bessuman.academia@gmail.com**
- Web: **www.proteinsimulationconsulting.com**

## License
MIT (see `LICENSE`).
