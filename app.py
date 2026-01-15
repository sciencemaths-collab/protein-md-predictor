#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tokenized Dynamics Physics v4 (state-of-the-art UI)

Streamlit dashboard with a unified "signal bus" that visually tracks:

preprocess → tokenize → train → predict → constrained decode → proof.

Upgrades in this build:
  • No duplicate Streamlit keys for Plotly.
  • Streaming uploads (large-file friendly) + URL download option.
  • Single, consistent session-state model (fixes type corruption on reruns).
  • Activity tab actually shows activity.
  • Optional interactive 3D viewer for predicted pause PDB (NGL).

Run:
  streamlit run app.py

Python: 3.11.8
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import time
import urllib.request

# Optional: requests provides nicer streaming + progress for huge URL ingests
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import plotly.graph_objects as go

# Ensure local package import works whether launched from repo root or elsewhere
ROOT = Path(__file__).resolve().parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tdphysics.utils import set_seed
from tdphysics.mdio import (
    make_trajectory_data,
    TrajectoryData,
    build_fixed_knn_edges,
    trajectory_to_distances,
)
from tdphysics.pipeline import (
    build_engine,
    compute_frame_energies,
    predict_pause_structures,
    export_pause_pdb,
)
from tdphysics.future import (
    build_engine_future,
    predict_future_movie,
    export_future_movie_pdb,
    BeamConfig,
)
from tdphysics.energy import EnergyWeights
from tdphysics.train import TrainConfig
from tdphysics.predict import RolloutConfig
from tdphysics.decode import DecodeWeights
from tdphysics.triplet import triplet_from_latent
from tdphysics.metrics import rmsd_kabsch, per_site_displacement


# -----------------------------
# Time unit helpers (UI convenience)
# -----------------------------

# Display units shown to users
TIME_UNITS_DISPLAY = ["ps", "ns", "µs", "ms", "s"]

# Internal unit strings used across the codebase
def _unit_internal(u: str) -> str:
    u = str(u).strip()
    return "us" if u in {"µs", "μs"} else u.lower()


# Convert a time value between units via an ns bridge.
_TO_NS = {"ps": 1e-3, "ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def _convert_time(value: float, from_unit: str, to_unit: str) -> float:
    fu = _unit_internal(from_unit)
    tu = _unit_internal(to_unit)
    if fu not in _TO_NS or tu not in _TO_NS:
        return float(value)
    return float(value) * _TO_NS[fu] / _TO_NS[tu]


# -----------------------------
# Page
# -----------------------------

st.set_page_config(page_title="Tokenized Dynamics Physics v4", layout="wide")


# -----------------------------
# Plotly helper: guaranteed unique keys
# -----------------------------

def pchart(fig: "go.Figure", *, container=None, **kwargs) -> None:
    if "_plotly_seq" not in st.session_state or not isinstance(st.session_state.get("_plotly_seq"), int):
        st.session_state["_plotly_seq"] = 0
    st.session_state["_plotly_seq"] += 1
    kwargs.pop("key", None)
    key = f"p_{st.session_state['_plotly_seq']}_{uuid4().hex[:8]}"
    if container is None:
        st.plotly_chart(fig, key=key, **kwargs)
    else:
        container.plotly_chart(fig, key=key, **kwargs)


# -----------------------------
# State
# -----------------------------

STAGES: List[Tuple[str, str]] = [
    ("Preprocess", "Distances / invariants"),
    ("Tokenize", "PCA + VQ codebook"),
    ("Train", "Physics-shaped losses"),
    ("Predict", "Energy-biased rollout"),
    ("Decode", "Constrained reconstruction"),
    ("Proof", "RMSD + displacement"),
]


def _init_state() -> None:
    # User-facing objects
    st.session_state.setdefault("seed", 0)
    st.session_state.setdefault("traj", None)
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("last_pred", None)

    # UI toggles
    st.session_state.setdefault("reduced_motion", False)
    st.session_state.setdefault("enable_3d", True)

    # Telemetry bus
    if "telemetry" not in st.session_state or not isinstance(st.session_state.get("telemetry"), dict):
        st.session_state["telemetry"] = {}
    t = st.session_state["telemetry"]
    t.setdefault("stage_idx", -1)
    t.setdefault("p", 0.0)
    t.setdefault("mode", "idle")
    t.setdefault("note", "Waiting for input.")
    t.setdefault("wave_y", [])

    # Metric series (used for bars / modulation)
    if "bus_series" not in st.session_state or not isinstance(st.session_state.get("bus_series"), dict):
        st.session_state["bus_series"] = {}
    s = st.session_state["bus_series"]
    for k in ("distance", "tokenize", "loss", "entropy", "decode", "disp", "movie"):
        if k not in s or not isinstance(s.get(k), list):
            s[k] = []

    # Activity
    if "activity" not in st.session_state or not isinstance(st.session_state.get("activity"), list):
        st.session_state["activity"] = []

    # internal throttles
    st.session_state.setdefault("_bus_last_render", 0.0)


_init_state()


def log(msg: str, kind: str = "info", data: Optional[dict] = None) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state["activity"].append({"t": ts, "kind": kind, "msg": str(msg), "data": data or {}})
    # Keep the latest status line visible on the bus
    st.session_state["telemetry"]["note"] = f"[{ts}] {msg}"


def _device_string() -> str:
    # best-effort GPU detection (optional)
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# -----------------------------
# Styling
# -----------------------------

def inject_css(reduced_motion: bool = False) -> None:
    """Inject the global CSS theme. Uses a placeholder to avoid f-string brace issues."""
    rm = "none" if reduced_motion else "pulse 1.3s ease-in-out infinite"
    css = r"""<style>
:root{
  --bg0: #0b0f14;
  --bg1: rgba(15,18,24,0.86);
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.09);
  --stroke: rgba(255,255,255,0.10);
  --stroke2: rgba(255,255,255,0.14);
  --txt: rgba(238,240,255,0.94);
  --muted: rgba(238,240,255,0.66);
  --a: rgba(255,85,115,1);
  --b: rgba(255,119,199,1);
  --g: rgba(137,255,171,1);
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 50% 10%, rgba(120,90,255,0.22), transparent 55%),
    radial-gradient(1000px 700px at 80% 30%, rgba(255,120,200,0.16), transparent 60%),
    radial-gradient(900px 600px at 10% 35%, rgba(80,220,255,0.14), transparent 60%),
    linear-gradient(180deg, #050611 0%, #080A1B 40%, #050611 100%),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.05) 0px, rgba(255,255,255,0.05) 1px, rgba(0,0,0,0) 1px, rgba(0,0,0,0) 42px),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0px, rgba(255,255,255,0.05) 1px, rgba(0,0,0,0) 1px, rgba(0,0,0,0) 42px);
  color: var(--txt);
}

.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }

/* hide Streamlit chrome */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* cards */
.tdp-card{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 14px;
  box-shadow: 0 12px 36px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}
.tdp-card-strong{
  background: var(--card2);
  border: 1px solid var(--stroke2);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 16px 44px rgba(0,0,0,0.42);
  backdrop-filter: blur(12px);
}

/* hover glow */
.tdp-card:hover, .tdp-card-strong:hover{
  border-color: rgba(255,255,255,0.20);
  box-shadow: 0 18px 58px rgba(0,0,0,0.55);
}

/* stage dot (idle / done / active) */
.dot{
  width:14px; height:14px;
  border-radius:50%; display:inline-block;
  background: rgba(255,255,255,0.22);
  border: 1px solid rgba(255,255,255,0.20);
  box-shadow: 0 0 0 rgba(0,0,0,0);
}
.dot.dot-done{
  background: var(--g);
  border-color: rgba(137,255,171,0.40);
  box-shadow: 0 0 12px rgba(137,255,171,0.55), 0 0 30px rgba(137,255,171,0.16);
}
.dot.dot-active{
  background: var(--a);
  border-color: rgba(255,85,115,0.55);
  box-shadow: 0 0 12px rgba(255,85,115,0.85), 0 0 30px rgba(255,85,115,0.18);
  animation: __RM__;
}
@keyframes pulse { 0%{ transform: scale(1.0); opacity: .85; } 50%{ transform: scale(1.25); opacity: 1.0; } 100%{ transform: scale(1.0); opacity: .85; } }


/* pipeline step cards */
.tdp-step{ position: relative; overflow: hidden; }
.tdp-step.tdp-step-done::before{ content: ""; position: absolute; inset: 0;
  background: radial-gradient(circle at 20% 0%, rgba(137,255,171,0.22), transparent 62%);
  opacity: 0.45; pointer-events: none; }
.tdp-step.tdp-step-active::before{ content: ""; position: absolute; inset: 0;
  background: radial-gradient(circle at 25% 0%, rgba(255,85,115,0.20), transparent 64%);
  opacity: 0.45; pointer-events: none; }
.tdp-step.tdp-step-done{ box-shadow: 0 0 22px rgba(137,255,171,0.12); }
.tdp-step.tdp-step-active{ box-shadow: 0 0 22px rgba(255,85,115,0.12); }

/* subtle checkmark (completed stages only) */
.tdp-chk{ margin-left: auto; font-size: 14px; font-weight: 900;
  color: rgba(137,255,171,0.92);
  opacity: 0; transform: scale(0.92);
  transition: opacity .18s ease, transform .18s ease;
  filter: drop-shadow(0 0 10px rgba(137,255,171,0.22));
}
.tdp-chk.show{ opacity: 0.95; transform: scale(1.0); }

/* sidebar */
[data-testid="stSidebar"] { background: rgba(11,12,26,0.92); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: var(--txt); }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid rgba(255,255,255,0.10); }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-bottom: none; border-radius: 12px 12px 0 0; padding: 10px 14px; }

/* hero */
.tdp-nav{
  position: sticky; top: 0; z-index: 99;
  display:flex; align-items:center; justify-content:space-between;
  padding: 10px 14px; margin: 6px 0 16px 0;
  background: rgba(10,12,16,0.62);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  backdrop-filter: blur(10px);
}
.tdp-nav .brand{ font-weight: 800; letter-spacing: .6px; color: rgba(255,255,255,0.92); }
.tdp-nav .links{ display:flex; gap:18px; align-items:center; }
.tdp-nav .link{ color: rgba(255,255,255,0.65); font-weight:700; font-size: 14px; }
.tdp-nav .link.active{ color: rgba(255,255,255,0.92); }
.tdp-hero .kicker{ color: rgba(255,255,255,0.72); font-weight: 800; letter-spacing: .6px; font-size: 12px; text-transform: uppercase; }
.tdp-hero .headline{ font-size: 44px; font-weight: 900; line-height: 1.05; color: rgba(255,255,255,0.95); margin-top: 6px; }
.tdp-hero .subhead{ margin-top: 10px; max-width: 980px; color: rgba(255,255,255,0.72); font-size: 16px; line-height: 1.45; }


/* ===== Widget readability overrides (fix white uploader / inputs) ===== */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(8,10,24,0.98), rgba(6,7,18,0.98)) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

div[data-testid="stFileUploader"] section{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}
div[data-testid="stFileUploader"] *{
  color: rgba(240,244,255,0.90) !important;
}
div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: rgba(240,244,255,0.92) !important;
  border-radius: 12px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
  background: rgba(255,255,255,0.06) !important;
  color: rgba(240,244,255,0.92) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
}

div[data-testid="stSelectbox"] div[role="combobox"]{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
}
div[data-testid="stSelectbox"] *{
  color: rgba(240,244,255,0.92) !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label{
  color: rgba(240,244,255,0.90) !important;
}




/* BaseWeb surfaces (Streamlit inputs/selects): HARD force dark + readable text */
section[data-testid="stSidebar"] div[data-baseweb="base-input"],
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="textarea"],
/* fallback (some Streamlit builds place widgets outside sidebar wrappers) */
div[data-baseweb="base-input"],
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="textarea"]{
  background-color: transparent !important;
}

section[data-testid="stSidebar"] div[data-baseweb="base-input"] > div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"] > div > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
section[data-testid="stSidebar"] div[data-baseweb="textarea"] > div,
section[data-testid="stSidebar"] div[data-baseweb="textarea"] > div > div,
div[data-baseweb="base-input"] > div,
div[data-baseweb="base-input"] > div > div,
div[data-baseweb="input"] > div,
div[data-baseweb="input"] > div > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="textarea"] > div > div{
  background-color: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
}

/* Actual input text (Safari needs -webkit-text-fill-color) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
input, textarea{
  color: rgba(240,244,255,0.95) !important;
  -webkit-text-fill-color: rgba(240,244,255,0.95) !important;
  caret-color: rgba(240,244,255,0.95) !important;
}

/* Selectbox visible value + arrow */
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input{
  color: rgba(240,244,255,0.95) !important;
  -webkit-text-fill-color: rgba(240,244,255,0.95) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
div[data-baseweb="select"] svg{
  color: rgba(240,244,255,0.85) !important;
  fill: rgba(240,244,255,0.85) !important;
}

/* NumberInput steppers */
div[data-testid="stNumberInput"] button{
  background-color: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
div[data-testid="stNumberInput"] button svg{
  color: rgba(240,244,255,0.85) !important;
  fill: rgba(240,244,255,0.85) !important;
}

/* Placeholder readability */
input::placeholder, textarea::placeholder{
  color: rgba(240,244,255,0.50) !important;
  -webkit-text-fill-color: rgba(240,244,255,0.50) !important;
}


</style>"""
    css = css.replace("__RM__", rm)
    st.markdown(css, unsafe_allow_html=True)




inject_css(bool(st.session_state.get("reduced_motion", False)))


# -----------------------------
# Hero + Stage bar
# -----------------------------

def hero() -> None:
    st.markdown(
        """
        <div class="tdp-hero">
          <div class="kicker">Tokenized Dynamics v4</div>
          <div class="headline">Molecular Dynamics and Structure Forecasting</div>
          <div class="subhead">
            Upload a trajectory, train the token engine, forecast future structures, then decode under constraints.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_bar(stage_idx: int, p: float, *, container=None) -> None:
    """Render the pipeline stepper cards with subtle stage lighting.

    - completed stages: green glow + checkmark
    - active stage: accent glow + pulse dot
    - idle stages: dim
    """
    stage_idx = int(stage_idx)
    p = float(max(0.0, min(1.0, p)))
    pct = int(p * 100)

    cards_html_parts: List[str] = []
    for i, (name, sub) in enumerate(STAGES):
        if stage_idx < 0:
            state = "idle"
        elif i < stage_idx:
            state = "done"
        elif i == stage_idx:
            state = "active"
        else:
            state = "idle"

        if state == "active":
            dot_class = "dot dot-active"
            op = "1.0"
            bg = "linear-gradient(135deg, rgba(255,85,115,0.12), rgba(255,255,255,0.04))"
            border = "1px solid rgba(255,85,115,0.28)"
            chk_class = "tdp-chk"
        elif state == "done":
            dot_class = "dot dot-done"
            op = "0.96"
            bg = "linear-gradient(135deg, rgba(137,255,171,0.12), rgba(255,255,255,0.04))"
            border = "1px solid rgba(137,255,171,0.20)"
            chk_class = "tdp-chk show"
        else:
            dot_class = "dot"
            op = "0.78"
            bg = "rgba(255,255,255,0.05)"
            border = "1px solid rgba(255,255,255,0.10)"
            chk_class = "tdp-chk"

        card_class = f"tdp-step tdp-step-{state}"

        card_html = (
            f'<div class="{card_class}" style="flex:1; min-width:170px; background:{bg}; '
            f'border:{border}; border-radius:14px; padding:12px 12px; opacity:{op};">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span class="{dot_class}"></span>'
            f'<div style="font-weight:900; font-size:15px;">{name}</div>'
            f'<span class="{chk_class}">&#10003;</span>'
            f'</div>'
            f'<div style="margin-top:6px; color:rgba(238,240,255,0.62); font-size:12px;">{sub}</div>'
            f'</div>'
        )
        cards_html_parts.append(card_html)

    cards_html = "".join(cards_html_parts)

    html = (
        f'<div class="tdp-card" style="padding:14px 14px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<div style="font-weight:900; font-size:16px;">Pipeline</div>'
        f'<div style="color:rgba(238,240,255,0.70); font-size:12px;">stage progress: <b>{pct}%</b></div>'
        f'</div>'
        f'<div style="margin-top:12px; display:flex; gap:12px; flex-wrap:wrap;">{cards_html}</div>'
        f'</div>'
    )
    (container or st).markdown(html, unsafe_allow_html=True)


def status_row() -> None:
    """Small status strip for device / trajectory / engine / prediction."""
    st.markdown(
        f"""<div class='tdp-card' style='margin-top:10px;'>
    <div style='display:flex; gap:12px; flex-wrap:wrap; align-items:stretch;'>
      <div style='flex:1; min-width:180px;'>
        <div style='color:rgba(238,240,255,0.70); font-weight:800; font-size:12px; text-transform:uppercase;'>Device</div>
        <div style='font-size:22px; font-weight:900;'>{_device_string()}</div>
      </div>
      <div style='flex:1; min-width:180px;'>
        <div style='color:rgba(238,240,255,0.70); font-weight:800; font-size:12px; text-transform:uppercase;'>Trajectory</div>
        <div style='font-size:22px; font-weight:900;'>{'Loaded' if st.session_state.get('traj') is not None else 'Empty'}</div>
      </div>
      <div style='flex:1; min-width:180px;'>
        <div style='color:rgba(238,240,255,0.70); font-weight:800; font-size:12px; text-transform:uppercase;'>Engine</div>
        <div style='font-size:22px; font-weight:900;'>{'Ready' if st.session_state.get('engine') is not None else 'None'}</div>
      </div>
      <div style='flex:1; min-width:180px;'>
        <div style='color:rgba(238,240,255,0.70); font-weight:800; font-size:12px; text-transform:uppercase;'>Prediction</div>
        <div style='font-size:22px; font-weight:900;'>{'Available' if st.session_state.get('last_pred') is not None else 'None'}</div>
      </div>
    </div>
    </div>""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Neon bus (canvas)
# -----------------------------

def _neon_banner_html(state: dict) -> str:
    cid = f"neo_{uuid4().hex[:10]}"
    state_json = json.dumps(state)
    return f"""
<div style="width:100%; height:250px; border-radius:22px; overflow:hidden;
            border:1px solid rgba(255,255,255,.08);
            background: radial-gradient(1200px 600px at 15% 10%, rgba(255,90,140,.10), rgba(0,0,0,0) 55%),
                        radial-gradient(900px 500px at 85% 0%, rgba(255,120,210,.08), rgba(0,0,0,0) 55%),
                        linear-gradient(180deg, rgba(10,12,18,0.98), rgba(8,10,15,0.98));
            position:relative;">
  <canvas id="{cid}" style="width:100%; height:100%; display:block;"></canvas>
  <script>
    const STATE = {state_json};
    const canvas = document.getElementById("{cid}");
    const ctx = canvas.getContext('2d');

    function resize(){{
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(r.width * devicePixelRatio));
      canvas.height = Math.max(1, Math.floor(r.height * devicePixelRatio));
    }}
    resize();
    new ResizeObserver(resize).observe(canvas);

    // deterministic RNG
    let seed = (STATE.seed ?? 0) >>> 0;
    function rng(){{ seed = (1664525 * seed + 1013904223) >>> 0; return (seed / 4294967296); }}

    // particles
    const P = [];
    function spawn(n){{
      for(let i=0;i<n;i++){{
        P.push({{ x:rng(), y:0.35 + 0.25*(rng()-0.5), vx:0.12*(rng()-0.5), vy:-0.10*rng(), life:0.6 + 0.8*rng(), s:0.7 + 1.8*rng() }});
      }}
    }}

    // fixed nodes along the wave
    const nodeXs = Array.from({{length:7}}, (_,i)=>0.10 + i*(0.80/6.0));

    // smooth bars (equalizer)
    const NB = 42;
    let bars = new Array(NB).fill(0);

    function grid(w,h){{
      ctx.save();
      ctx.globalAlpha = 0.10;
      ctx.lineWidth = 1;
      const step = Math.max(26, Math.floor(w/28));
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      for(let x=0; x<=w; x+=step){{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }}
      for(let y=0; y<=h; y+=step){{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }}
      ctx.restore();
    }}

    function glowWave(w,h,t){{
      const mid = Math.floor(h*0.45);
      const p = Math.max(0, Math.min(1, (STATE.p ?? 0)));
      const mode = (STATE.mode ?? "idle");
      const speed = (mode.includes("predict")) ? 1.9 : (mode.includes("train")) ? 1.3 : (mode.includes("decode")) ? 1.6 : 1.0;
      const freq  = (mode.includes("predict")) ? 0.013 : (mode.includes("train")) ? 0.010 : (mode.includes("decode")) ? 0.011 : 0.009;
      const ampBase = 16 + 22*p;

      const waveY = Array.isArray(STATE.wave_y) ? STATE.wave_y : [];
      const haveTele = waveY.length > 8;

      const N = 620;
      const pts = new Array(N);
      for(let i=0;i<N;i++){{
        const u = i/(N-1);
        const x = u*w;
        const carrier = Math.sin((x*freq) + t*speed) + 0.35*Math.sin((x*freq*1.9) - t*0.9);
        let mod = 1.0;
        if(haveTele){{
          const j = Math.floor(u*(waveY.length-1));
          const v = waveY[j] ?? 0;
          mod = 0.65 + 0.60*Math.min(1.0, Math.abs(v));
        }}
        const y = mid + carrier * ampBase * mod;
        pts[i] = [x,y];
      }}

      const accentA = "rgba(255,90,140,0.95)";
      const accentB = "rgba(255,120,210,0.92)";

      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const grad = ctx.createLinearGradient(0,0,w,0);
      grad.addColorStop(0.00, accentA);
      grad.addColorStop(0.50, accentB);
      grad.addColorStop(1.00, accentA);

      // outer glow
      ctx.strokeStyle = grad;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 18;
      ctx.shadowBlur = 30;
      ctx.shadowColor = accentB;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for(let i=1;i<N;i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();

      // inner glow
      ctx.globalAlpha = 0.32;
      ctx.lineWidth = 10;
      ctx.shadowBlur = 20;
      ctx.shadowColor = accentA;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for(let i=1;i<N;i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();

      // white core
      ctx.globalAlpha = 0.95;
      ctx.lineWidth = 2.1;
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255,255,255,0.92)";
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for(let i=1;i<N;i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();

      // nodes
      for(const ux of nodeXs){{
        const ix = Math.floor(ux*(N-1));
        const x = pts[ix][0], y = pts[ix][1];
        const pulse = 0.75 + 0.35*Math.sin(t*2.2 + ux*10.0);
        const r = 6.2 * pulse;
        const rg = ctx.createRadialGradient(x,y,0,x,y, r*2.8);
        rg.addColorStop(0.00, "rgba(255,140,90,0.95)");
        rg.addColorStop(0.35, "rgba(255,90,140,0.65)");
        rg.addColorStop(1.00, "rgba(255,90,140,0.0)");
        ctx.fillStyle = rg;
        ctx.globalAlpha = 0.95;
        ctx.beginPath(); ctx.arc(x,y,r*2.2,0,Math.PI*2); ctx.fill();

        ctx.lineWidth = 1.2;
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.stroke();
      }}

      ctx.restore();
      return pts;
    }}

    function drawBars(w,h,pts){{
      const baseY = Math.floor(h*0.78);
      const maxH = Math.floor(h*0.16);
      const left = Math.floor(w*0.06);
      const right = Math.floor(w*0.94);
      const width = right-left;
      const gap = width/NB;
      const bw = Math.max(2, Math.floor(gap*0.55));

      const haveTeleBars = Array.isArray(STATE.bars) && STATE.bars.length>4;

      for(let i=0;i<NB;i++){{
        let tgt = 0.25 + 0.75*rng();
        if(haveTeleBars){{
          const j = Math.floor(i*(STATE.bars.length-1)/(NB-1));
          const v = STATE.bars[j] ?? 0;
          tgt = Math.min(1.0, Math.max(0.0, Math.abs(v)));
        }} else {{
          const u = i/(NB-1);
          const ix = Math.floor(u*(pts.length-1));
          const y = pts[ix][1];
          const dy = Math.abs(y - h*0.45)/(h*0.25);
          tgt = Math.min(1.0, 0.15 + 0.85*dy);
        }}

        bars[i] = 0.82*bars[i] + 0.18*tgt;
        const x = left + i*gap;
        const hh = bars[i]*maxH;

        ctx.save();
        ctx.globalCompositeOperation = "lighter";

        ctx.shadowBlur = 18;
        ctx.shadowColor = "rgba(255,90,140,0.95)";
        const g = ctx.createLinearGradient(0, baseY-hh, 0, baseY);
        g.addColorStop(0.0, "rgba(255,140,90,0.85)");
        g.addColorStop(0.6, "rgba(255,120,210,0.92)");
        g.addColorStop(1.0, "rgba(255,90,140,0.12)");
        ctx.fillStyle = g;
        ctx.globalAlpha = 0.55;
        ctx.fillRect(x, baseY-hh, bw, hh);

        ctx.shadowBlur = 0;
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = "rgba(255,255,255,0.16)";
        ctx.fillRect(x, baseY-hh, bw, hh);

        ctx.restore();
      }}
    }}

    function draw(tms){{
      const t = tms/1000;
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0,0,w,h);

      // nebula bloom
      ctx.save();
      const g = ctx.createRadialGradient(w*0.18,h*0.25,0,w*0.18,h*0.25,w*1.05);
      g.addColorStop(0,"rgba(255,90,140,0.10)");
      g.addColorStop(0.55,"rgba(255,90,140,0.0)");
      ctx.fillStyle = g;
      ctx.fillRect(0,0,w,h);
      ctx.restore();

      grid(w,h);

      // scanline
      ctx.save();
      const sy = (t*42) % h;
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(0, sy, w, 2*devicePixelRatio);
      ctx.restore();

      const p = Math.max(0, Math.min(1, (STATE.p ?? 0)));
      spawn(Math.floor(1 + 3*p));
      const pts = glowWave(w,h,t);
      drawBars(w,h,pts);

      // particles
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      for(let i=P.length-1;i>=0;i--){{
        const q = P[i];
        q.life -= 0.012;
        q.x += q.vx*0.010;
        q.y += q.vy*0.010;
        if(q.life<=0){{ P.splice(i,1); continue; }}
        const x = q.x*w;
        const y = (0.45*q.y + 0.45)*h;
        const a = Math.max(0, Math.min(1, q.life));
        ctx.globalAlpha = 0.35*a;
        ctx.fillStyle = "rgba(255,160,110,0.9)";
        ctx.beginPath(); ctx.arc(x,y,q.s*devicePixelRatio,0,Math.PI*2); ctx.fill();
      }}
      ctx.restore();

      requestAnimationFrame(draw);
    }}

    requestAnimationFrame(draw);
  </script>
</div>
"""


def _carrier_wave(p: float, n: int = 260, base_freq: float = 6.0) -> List[float]:
    amp = 0.15 + 0.85 * float(max(0.0, min(1.0, p)))
    xs = np.linspace(0, 1, n, dtype=np.float32)
    y = amp * np.sin(2 * math.pi * base_freq * xs) + 0.08 * np.sin(2 * math.pi * (base_freq * 2.7) * xs)
    return y.astype(np.float32).tolist()



def render_residue_network_ring(
    widget_id: str,
    intensity: float = 0.6,
    stage: int = 0,
    size: int = 220,
    *,
    nodes: Optional[List[float]] = None,
    edges_i: Optional[List[int]] = None,
    edges_j: Optional[List[int]] = None,
    edges_w: Optional[List[float]] = None,
) -> None:
    """Circular residue-network ring (HTML canvas).

    If nodes/edges_* are provided, the glow is *data-driven*:
      - node glow ~ per-node activity (e.g., sum |Δd| on incident edges)
      - edge glow ~ per-edge |Δd|
    Otherwise it falls back to a deterministic "alive" animation.
    """
    intensity = max(0.0, min(1.0, float(intensity)))
    stage = int(stage)

    has_data = (
        isinstance(nodes, list)
        and isinstance(edges_i, list)
        and isinstance(edges_j, list)
        and isinstance(edges_w, list)
        and (len(nodes) > 0)
        and (len(edges_i) == len(edges_j) == len(edges_w))
    )

    # Keep payload small and safe for HTML embedding
    nodes_js = json.dumps([float(x) for x in (nodes or [])][:256])
    ei_js = json.dumps([int(x) for x in (edges_i or [])][:1200])
    ej_js = json.dumps([int(x) for x in (edges_j or [])][:1200])
    ew_js = json.dumps([float(x) for x in (edges_w or [])][:1200])

    html = r"""
<div style="width: __SIZE__px; height: __SIZE__px; margin: 0 auto;">
  <canvas id="__ID__" width="__SIZE__" height="__SIZE__"
    style="width: __SIZE__px; height: __SIZE__px; border-radius: 999px;
           background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.06), rgba(0,0,0,0) 55%),
                       rgba(255,255,255,0.04);
           border: 1px solid rgba(255,255,255,0.10);
           box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset;">
  </canvas>
</div>

<script>
(() => {
  const canvas = document.getElementById("__ID__");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const W = canvas.width, H = canvas.height;
  const cx = W/2, cy = H/2;

  const intensity = __INT__;
  const stage = __STAGE__;
  const HAS_DATA = __HASDATA__;

  const NODES = __NODES__;
  const EI = __EI__;
  const EJ = __EJ__;
  const EW = __EW__;

  const K = HAS_DATA ? (NODES.length|0) : 64;
  const R = Math.min(W,H) * 0.36;
  const jitter = Math.min(W,H) * 0.06;

  // deterministic-ish seed from stage
  function mulberry32(a){return function(){var t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
  const rand = mulberry32(12345 + stage*999);

  // fixed ring nodes (each corresponds to a residue index in your sampled set)
  const pts = Array.from({length: K}, (_,i)=>({
    a0: (i / Math.max(1, K)) * Math.PI*2,
    phase: rand()*Math.PI*2,
    drift: (0.15 + rand()*0.55) * (0.35 + intensity*1.4),
    r: R + (rand()*2-1)*jitter,
    s: 0.7 + rand()*0.9,
  }));

  let t0 = performance.now();
  let rot = 0;

  function hueFor(i){
    // unique "signature" per node
    return ( (i * 360 / Math.max(1,K)) + (stage*21) ) % 360;
  }

  function draw(t){
    const dt = (t - t0) * 0.001;
    t0 = t;
    rot += dt * (0.12 + 0.25*intensity);

    ctx.clearRect(0,0,W,H);

    // soft outer ring
    ctx.beginPath();
    ctx.arc(cx,cy,R,0,Math.PI*2);
    ctx.strokeStyle = `rgba(240,244,255,${0.10 + intensity*0.18})`;
    ctx.lineWidth = 1.0;
    ctx.stroke();

    // compute positions + per-node glow
    const pos = pts.map((p,i)=>{
      const pulse = 0.55 + 0.45*Math.sin((t*0.002) + p.phase);
      const rr = p.r * (0.92 + 0.10*pulse*intensity);
      const a = p.a0 + rot + dt*0.0;
      const x = cx + Math.cos(a)*rr;
      const y = cy + Math.sin(a)*rr;

      let g = (0.25 + 0.75*pulse) * (0.25 + intensity);
      if (HAS_DATA && i < NODES.length){
        // data-driven activity wins (with a tiny breathing motion)
        const v = Math.max(0, Math.min(1, NODES[i]));
        g = (0.12 + 0.88*v) * (0.55 + 0.45*pulse) * (0.35 + 0.90*intensity);
      }
      return {x, y, glow: g, pulse};
    });

    // network edges
    if (HAS_DATA && EI.length === EJ.length && EI.length === EW.length){
      for (let k=0; k<EI.length; k++){
        const i = EI[k]|0, j = EJ[k]|0;
        if (i<0 || j<0 || i>=K || j>=K) continue;
        const w = Math.max(0, Math.min(1, EW[k]));
        if (w <= 1e-4) continue;

        const a = (0.04 + 0.22*intensity) * (0.20 + 0.80*w);
        const hi = hueFor(i), hj = hueFor(j);
        const h = (hi + hj) * 0.5;

        ctx.beginPath();
        ctx.moveTo(pos[i].x, pos[i].y);
        ctx.lineTo(pos[j].x, pos[j].y);
        ctx.strokeStyle = `hsla(${h}, 95%, 70%, ${a})`;
        ctx.lineWidth = 1.0;
        ctx.stroke();
      }
    } else {
      // fallback faint edges (nearby only)
      for (let i=0; i<K; i++){
        for (let j=i+1; j<K; j++){
          const dx = pos[i].x - pos[j].x;
          const dy = pos[i].y - pos[j].y;
          const d2 = dx*dx + dy*dy;
          if (d2 < (W*W)*0.03){
            const a = (0.06 + 0.16*intensity) * (1.0 - d2/((W*W)*0.03));
            ctx.beginPath();
            ctx.moveTo(pos[i].x, pos[i].y);
            ctx.lineTo(pos[j].x, pos[j].y);
            ctx.strokeStyle = `rgba(160,210,255,${a})`;
            ctx.lineWidth = 1.0;
            ctx.stroke();
          }
        }
      }
    }

    // particles (unique light signature per node)
    for (let i=0; i<K; i++){
      const g = pos[i].glow;
      const r = (1.2 + 2.2*pts[i].s) * (0.85 + 0.9*intensity) * (0.55 + 0.85*g);
      const h = hueFor(i);

      const grad = ctx.createRadialGradient(pos[i].x, pos[i].y, 0, pos[i].x, pos[i].y, r*3.2);
      grad.addColorStop(0, `hsla(${h}, 95%, 72%, ${0.08 + 0.30*g})`);
      grad.addColorStop(0.5, `hsla(${(h+160)%360}, 95%, 70%, ${0.06 + 0.22*g})`);
      grad.addColorStop(1, `rgba(0,0,0,0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pos[i].x, pos[i].y, r*3.0, 0, Math.PI*2);
      ctx.fill();

      ctx.fillStyle = `rgba(240,244,255,${0.28 + 0.62*g})`;
      ctx.beginPath();
      ctx.arc(pos[i].x, pos[i].y, r, 0, Math.PI*2);
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
})();
</script>
"""
    html = (
        html.replace("__ID__", widget_id)
        .replace("__INT__", f"{intensity:.4f}")
        .replace("__STAGE__", str(stage))
        .replace("__SIZE__", str(int(size)))
        .replace("__HASDATA__", "true" if has_data else "false")
        .replace("__NODES__", nodes_js)
        .replace("__EI__", ei_js)
        .replace("__EJ__", ej_js)
        .replace("__EW__", ew_js)
    )

    components.html(html, height=size + 10)

def bus_render(*, container=None) -> None:
    t = st.session_state["telemetry"]
    s = st.session_state["bus_series"]

    # bars: first non-empty series gets used
    bars: List[float] = []
    for key in ("loss", "entropy", "decode", "distance", "disp"):
        arr = s.get(key, [])
        if isinstance(arr, list) and len(arr) > 0:
            tail = [abs(float(x)) for x in arr[-42:]]
            mx = max(tail) if tail else 1.0
            mx = mx if mx > 1e-12 else 1.0
            bars = [x / mx for x in tail]
            break

    state = {
        "seed": int(st.session_state.get("seed", 0)),
        "stage_idx": int(t.get("stage_idx", -1)),
        "p": float(max(0.0, min(1.0, float(t.get("p", 0.0))))),
        "mode": str(t.get("mode", "idle")),
        "note": str(t.get("note", "")),
        "wave_y": list(t.get("wave_y", []))[-420:],
        "bars": bars,
        "ring_nodes": t.get("ring_nodes", []),
        "ring_ei": t.get("ring_ei", []),
        "ring_ej": t.get("ring_ej", []),
        "ring_ew": t.get("ring_ew", []),
    }

    ring_intensity = 0.18 + 0.82 * float(state["p"])
    ring_stage = int(state["stage_idx"])
    ring_size = 220

    def _draw() -> None:
        c1, c2 = st.columns([5, 2], gap="small")
        with c1:
            components.html(_neon_banner_html(state), height=255, scrolling=False)
        with c2:
            render_residue_network_ring("residue_ring", intensity=ring_intensity, stage=ring_stage, size=ring_size, nodes=state.get("ring_nodes"), edges_i=state.get("ring_ei"), edges_j=state.get("ring_ej"), edges_w=state.get("ring_ew"))

    if container is None:
        _draw()
    else:
        # Update inside a placeholder so the banner refreshes during long runs.
        container.empty()
        with container:
            _draw()




def _set_bus(stage_idx: int, p: float, mode: str, wave_y: List[float], note: str) -> None:
    t = st.session_state["telemetry"]
    t["stage_idx"] = int(stage_idx)
    t["p"] = float(max(0.0, min(1.0, p)))
    t["mode"] = str(mode)
    t["wave_y"] = list(wave_y)[-420:]
    t["note"] = str(note)


def make_progress_cb(*, render_hook=None) -> Any:
    s = st.session_state["bus_series"]

    def _push(series_key: str, value: float, maxlen: int = 1500) -> None:
        arr = s.get(series_key, [])
        arr.append(float(value))
        if len(arr) > maxlen:
            arr[:] = arr[-maxlen:]
        s[series_key] = arr

    def cb(event: str, payload: Optional[dict] = None) -> None:
        payload = payload or {}
        e = str(event)
        t = st.session_state.get("telemetry", {})
        # If backend emits ring signals (data-driven residue network), store them for bus_render.
        if isinstance(payload, dict) and ("ring_nodes" in payload):
            try:
                t["ring_nodes"] = payload.get("ring_nodes", t.get("ring_nodes", []))
                t["ring_ei"] = payload.get("ring_ei", t.get("ring_ei", []))
                t["ring_ej"] = payload.get("ring_ej", t.get("ring_ej", []))
                t["ring_ew"] = payload.get("ring_ew", t.get("ring_ew", []))
            except Exception:
                pass
        if e.startswith("pipeline."):
            e = e[len("pipeline.") :]

        p = float(payload.get("p", st.session_state["telemetry"].get("p", 0.0)))
        stage_idx = int(st.session_state["telemetry"].get("stage_idx", -1))
        mode = str(st.session_state["telemetry"].get("mode", "idle"))
        note = str(st.session_state["telemetry"].get("note", ""))

        # PREPROCESS
        if e.startswith("preprocess"):
            stage_idx = 0
            mode = "preprocess"
            note = "Preprocess: stream frames → align → fixed kNN → invariant distances."
            if "edge0" in payload:
                _push("distance", float(payload["edge0"]))
            wave = s["distance"][-420:]

        # TOKENIZE
        elif e.startswith("tokenize"):
            stage_idx = 1
            mode = "tokenize"
            note = "Tokenize: PCA compression + VQ (k-means) codebook → discrete tokens."
            if e in {"tokenize.pca", "tokenize.kmeans"}:
                ch = float(payload.get("chunk", 0.0))
                _push("tokenize", math.sin(ch * 0.85) * (0.25 + 0.75 * p))
            wave = s["tokenize"][-420:]

        # TRAIN (includes energy labeling in this implementation)
        elif e.startswith("energy") or e.startswith("train"):
            stage_idx = 2
            mode = "train"
            note = "Train: transformer learns multi-timescale token dynamics with physics regularizers."
            if e == "train.epoch" and "loss_ce" in payload:
                _push("loss", float(payload.get("loss_ce", 0.0)))
            wave = s["loss"][-420:] if len(s["loss"]) else s["tokenize"][-420:]

        # PREDICT
        elif e.startswith("predict"):
            stage_idx = 3
            mode = "predict"
            note = "Predict: rollout tokens with energy-biased sampling (stable long jumps)."
            if e == "predict.step" and "entropy" in payload:
                _push("entropy", float(payload.get("entropy", 0.0)))
            wave = s["entropy"][-420:]

        # DECODE
        elif e.startswith("decode"):
            stage_idx = 4
            mode = "decode"
            note = "Decode: reconstruct coordinates by constrained optimization (fit+bond+repel+smooth)."
            if e == "decode.step" and "loss" in payload:
                _push("decode", float(payload.get("loss", 0.0)))
            wave = s["decode"][-420:]

        # MOVIE (multi-frame decode)
        elif e.startswith("movie"):
            stage_idx = 4
            mode = "movie"
            note = "Movie: rollout → decode multiple frames into a playable trajectory (multi-model PDB)."
            # Light visual pulse for the bus
            _push("movie", math.sin((p * 12.0) + (stage_idx + 1)) * (0.25 + 0.75 * p))
            wave = s["movie"][-420:]

        # PROOF
        elif e.startswith("proof"):
            stage_idx = 5
            mode = "proof"
            note = "Proof: quantify difference between last frame and predicted pause (Kabsch RMSD)."
            disp = payload.get("disp", None)
            if isinstance(disp, (list, tuple)) and len(disp) > 0:
                s["disp"] = [float(v) for v in disp][-420:]
            wave = s["disp"][-420:]

        else:
            wave = st.session_state["telemetry"].get("wave_y") or _carrier_wave(p)

        _set_bus(stage_idx, p, mode, wave, note)

        # Throttle repaint; optionally refresh the pipeline/bus live during long runs.
        now = time.time()
        force = e.endswith(".done") or e in {"train.epoch", "predict.step", "decode.step", "proof.rmsd"}
        if force or (now - float(st.session_state.get("_bus_last_render", 0.0)) > 0.10):
            st.session_state["_bus_last_render"] = now
            if callable(render_hook):
                try:
                    render_hook()
                except Exception:
                    # Never let UI refresh errors break the pipeline.
                    pass

    return cb


# -----------------------------
# Upload helpers (streaming + URL)
# -----------------------------

def _suffix_from_name(name: str, fallback: str) -> str:
    name = (name or "").strip()
    if "." in name:
        ext = "." + name.split(".")[-1].lower()
        if len(ext) <= 8:
            return ext
    return fallback


def save_uploaded_to_temp(upload: Any, *, fallback_suffix: str) -> str:
    """Save Streamlit UploadedFile to disk without loading whole file in RAM."""
    suffix = _suffix_from_name(getattr(upload, "name", ""), fallback_suffix)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # UploadedFile behaves like a file-like object
    try:
        upload.seek(0)
    except Exception:
        pass
    # chunked copy
    while True:
        chunk = upload.read(1024 * 1024)
        if not chunk:
            break
        tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def download_url_to_temp(url: str, *, fallback_suffix: str, chunk_mb: int = 8, progress=None) -> str:
    """Download URL → temp file, streaming to disk (safe for huge inputs).

    Uses requests (if available) for progress via Content-Length; otherwise falls back to urllib.
    progress: optional callable p in [0, 1]
    """
    url = (url or "").strip()
    if not url:
        raise RuntimeError("Empty URL")

    # Guess suffix from URL path (ignoring query/fragment), preserve ext for MDAnalysis
    clean_name = url.split("?")[0].split("#")[0]
    suffix = _suffix_from_name(clean_name, fallback_suffix)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    chunk = max(1, int(chunk_mb)) * 1024 * 1024

    def _urllib_download() -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "TDPhysics/1.0"})
        with urllib.request.urlopen(req) as r:
            total = int(getattr(r, "length", 0) or 0)
            done = 0
            while True:
                part = r.read(chunk)
                if not part:
                    break
                tmp.write(part)
                done += len(part)
                if progress and total > 0:
                    progress(min(1.0, done / total))

    if requests is not None:
        try:
            headers = {"User-Agent": "TDPhysics/1.0"}
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:  # type: ignore[attr-defined]
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                done = 0
                for part in r.iter_content(chunk_size=chunk):
                    if not part:
                        continue
                    tmp.write(part)
                    done += len(part)
                    if progress and total > 0:
                        progress(min(1.0, done / total))
        except Exception:
            # requests can fail for some hosts; fall back to urllib
            try:
                tmp.seek(0)
                tmp.truncate(0)
            except Exception:
                pass
            _urllib_download()
    else:
        _urllib_download()

    tmp.flush()
    tmp.close()
    return tmp.name


# -----------------------------
# Synthetic demo
# -----------------------------


def build_ring_spec(
    engine,
    n_nodes: int = 64,
    knn: int = 3,
    cutoff: float = 8.0,
    max_edges: int = 220,
) -> dict:
    """Build a small residue-network spec for the circular ring visual.

    Returns a dict with:
      - node_idx: absolute residue indices into X (len K)
      - edges_node: (M,2) endpoints in [0..K-1]
      - edges_abs: (M,2) endpoints in absolute residue indices
      - node_resids/node_resnames: metadata for labeling (optional)
    """
    X_ref = np.asarray(engine.traj.X[-1], dtype=np.float32)
    N = int(X_ref.shape[0])
    n_nodes = int(max(8, min(int(n_nodes), N)))
    if N <= n_nodes:
        node_idx = np.arange(N, dtype=int)
    else:
        node_idx = np.unique(np.linspace(0, N - 1, n_nodes, dtype=int))
    K = int(node_idx.shape[0])

    coords = X_ref[node_idx, :]
    # Pairwise distances among selected nodes (KxK)
    dif = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(dif, axis=2).astype(np.float32)
    np.fill_diagonal(D, np.inf)

    edges = set()

    # kNN edges
    knn = int(max(1, min(int(knn), K - 1)))
    for i in range(K):
        js = np.argsort(D[i])[:knn]
        for j in js:
            a, b = (i, int(j))
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))

    # contact edges under cutoff
    if float(cutoff) > 0:
        cut = float(cutoff)
        ii, jj = np.where(D < cut)
        for a, b in zip(ii.tolist(), jj.tolist()):
            a = int(a); b = int(b)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))

    edges = sorted(edges)
    # Cap edges by shortest distances for clarity/perf
    if len(edges) > int(max_edges):
        dlist = [(float(D[a, b]), a, b) for (a, b) in edges]
        dlist.sort(key=lambda x: x[0])
        edges = [(a, b) for (_, a, b) in dlist[: int(max_edges)]]

    edges_node = np.asarray(edges, dtype=int) if len(edges) > 0 else np.zeros((0, 2), dtype=int)
    edges_abs = node_idx[edges_node] if edges_node.size else np.zeros((0, 2), dtype=int)

    resids = getattr(engine.traj, "resids", None)
    resnames = getattr(engine.traj, "resnames", None)
    node_resids = (np.asarray(resids, dtype=int)[node_idx].tolist() if resids is not None else [])
    node_resnames = (np.asarray(resnames)[node_idx].tolist() if resnames is not None else [])

    return {
        "node_idx": node_idx.astype(int).tolist(),
        "edges_node": edges_node.astype(int).tolist(),
        "edges_abs": edges_abs.astype(int).tolist(),
        "node_resids": node_resids,
        "node_resnames": node_resnames,
    }


def synthetic_demo(progress_cb=None, T: int = 800, N: int = 60, time_unit: str = "ns") -> TrajectoryData:
    if progress_cb:
        progress_cb("preprocess.start", {"p": 0.0})

    t = np.linspace(0, 30 * np.pi, T).astype(np.float32)
    base = np.stack([np.linspace(0, 80, N), np.zeros(N), np.zeros(N)], axis=-1).astype(np.float32)
    X = np.zeros((T, N, 3), dtype=np.float32)

    for k in range(T):
        phase = t[k]
        wiggle = np.stack(
            [
                np.zeros(N),
                2.0 * np.sin(0.3 * phase + np.linspace(0, 3 * np.pi, N)),
                1.0 * np.cos(0.2 * phase + np.linspace(0, 2 * np.pi, N)),
            ],
            axis=-1,
        ).astype(np.float32)
        noise = 0.15 * np.random.randn(N, 3).astype(np.float32)
        X[k] = base + wiggle + noise
        if progress_cb and (k % max(1, T // 80) == 0 or k == T - 1):
            progress_cb("preprocess.load", {"p": (k + 1) / T, "frame": k + 1, "n_frames": T, "edge0": float(X[k, 0, 0])})

    edges = build_fixed_knn_edges(X[0], k=int(st.session_state.get("k_nn", 8)), include_backbone=True)
    d = trajectory_to_distances(X, edges, progress_cb=progress_cb)

    resids = np.arange(1, N + 1, dtype=int)
    resnames = np.array(["GLY"] * N, dtype=object)
    # Keep demo dt consistent with the chosen unit (dt=0.05 in that unit)
    return TrajectoryData(resids=resids, resnames=resnames, X_ref=X[0], X=X, edges=edges, d=d, dt=0.05, time_unit=time_unit)


def build_traj(
    *,
    demo: bool,
    upload_mode: str,
    top_upload: Any,
    traj_upload: Any,
    top_url: str,
    traj_url: str,
    selection: str,
    stride: int,
    time_total: Optional[float],
    time_unit: str,
    k_nn: int,
    progress_cb=None,
) -> TrajectoryData:
    if demo:
        return synthetic_demo(progress_cb=progress_cb, time_unit=time_unit)

    if upload_mode == "Upload":
        if top_upload is None:
            raise RuntimeError("Upload topology file.")
        top_path = save_uploaded_to_temp(top_upload, fallback_suffix=".pdb")
        traj_path = save_uploaded_to_temp(traj_upload, fallback_suffix=".xtc") if traj_upload else None
    else:
        if not top_url:
            raise RuntimeError("Provide topology URL.")

        chunk_mb = int(st.session_state.get("url_chunk_mb", 8))

        # Stream-download to disk (supports very large files). Progress appears in the sidebar.
        pb = st.sidebar.progress(0.0)
        top_path = download_url_to_temp(top_url, fallback_suffix=".pdb", chunk_mb=chunk_mb, progress=pb.progress)
        pb.empty()

        traj_path = None
        if traj_url:
            pb2 = st.sidebar.progress(0.0)
            traj_path = download_url_to_temp(traj_url, fallback_suffix=".xtc", chunk_mb=chunk_mb, progress=pb2.progress)
            pb2.empty()

    return make_trajectory_data(
        top_path,
        traj_path,
        selection=selection,
        stride=int(stride),
        time_total=time_total,
        time_unit=time_unit,
        k=int(k_nn),
        align=True,
        # Forward the caller-provided progress callback (used by the bus/stepper).
        progress_cb=progress_cb,
    )


# -----------------------------
# 3D viewer (optional)
# -----------------------------

def render_ngl(pdb_bytes: bytes, *, height: int = 520) -> None:
    """Interactive 3D PDB view with NGL (CDN)."""
    b64 = base64.b64encode(pdb_bytes).decode("ascii")
    cid = f"ngl_{uuid4().hex[:8]}"
    html = f"""
<div id="{cid}" style="width:100%; height:{height}px; border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,0.10);"></div>
<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
<script>
  (function(){{
    const el = document.getElementById("{cid}");
    const stage = new NGL.Stage(el, {{ backgroundColor: "rgba(0,0,0,0)" }});
    function b64ToBlob(b64Data, contentType){{
      contentType = contentType || '';
      const byteCharacters = atob(b64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++){{ byteNumbers[i] = byteCharacters.charCodeAt(i); }}
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], {{type: contentType}});
    }}
    const blob = b64ToBlob("{b64}", "text/plain");
    stage.loadFile(blob, {{ ext: "pdb" }}).then(function(o){{
      o.addRepresentation("cartoon", {{ color: "chainname" }});
      o.addRepresentation("ball+stick", {{ sele: "backbone" }});
      o.autoView();
    }});
    window.addEventListener("resize", () => stage.handleResize(), false);
  }})();
</script>
"""
    components.html(html, height=height + 15, scrolling=False)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Controls")

    st.session_state["seed"] = int(st.number_input("Seed", value=int(st.session_state.get("seed", 0)), step=1))
    set_seed(int(st.session_state["seed"]))

    st.session_state["reduced_motion"] = bool(st.toggle("Reduced motion", value=bool(st.session_state.get("reduced_motion", False))))
    st.session_state["enable_3d"] = bool(st.toggle("Interactive 3D viewer", value=bool(st.session_state.get("enable_3d", True))))

    st.divider()
    demo = st.checkbox("Use synthetic demo", value=False, help="Runs without uploads. Great for UI/flow checks.")

    upload_mode = st.radio("Input mode", ["Upload", "URL"], horizontal=True, disabled=demo)
    if upload_mode == "URL":
        st.caption(
            "URL mode is recommended for very large trajectories. The app streams downloads to disk (not RAM). "
            "Note: Streamlit Community Cloud still has hard platform limits for uploads and local storage."
        )
        st.session_state["url_chunk_mb"] = int(
            st.slider(
                "URL download chunk (MB)",
                min_value=1,
                max_value=64,
                value=int(st.session_state.get("url_chunk_mb", 8)),
                step=1,
            )
        )

    top_upload = traj_upload = None
    top_url = traj_url = ""
    if not demo:
        if upload_mode == "Upload":
            top_upload = st.file_uploader("Topology", type=["pdb", "gro", "psf", "tpr"], help="PDB recommended.")
            traj_upload = st.file_uploader("Trajectory", type=["xtc", "dcd", "trr", "nc"], help="Optional for single-frame topology-only.")
        else:
            top_url = st.text_input("Topology URL", value="", placeholder="https://.../protein.pdb")
            traj_url = st.text_input("Trajectory URL (optional)", value="", placeholder="https://.../traj.xtc")

    selection = st.text_input("Atom selection", value="protein and name CA", help="MDAnalysis selection string.")
    stride = int(st.number_input("Stride", min_value=1, value=1, step=1))

    time_unit_display = st.selectbox(
        "Trajectory time unit",
        TIME_UNITS_DISPLAY,
        index=int(st.session_state.get("time_unit_idx", 1)),
        help="Used for dt display and time inputs. Internally converted as needed.",
    )
    st.session_state["time_unit_idx"] = int(TIME_UNITS_DISPLAY.index(time_unit_display))
    time_unit = _unit_internal(time_unit_display)

    time_total_val = float(
        st.number_input(
            f"Total trajectory time ({time_unit_display})",
            min_value=0.0,
            value=float(st.session_state.get("time_total_val", 0.0)),
            step=10.0,
            help="0 = auto from trajectory if available.",
        )
    )
    st.session_state["time_total_val"] = float(time_total_val)
    time_total = None if time_total_val == 0.0 else time_total_val

    st.divider()
    st.subheader("Tokenization")
    m = int(st.number_input("Latent dim (PCA)", min_value=2, value=48, step=1))
    K = int(st.number_input("Codebook size (tokens)", min_value=8, value=256, step=8))
    k_nn = int(st.number_input("kNN edges per node", min_value=2, value=8, step=1))
    st.session_state["k_nn"] = k_nn

    st.divider()
    st.subheader("Training")
    epochs = int(st.number_input("Epochs", min_value=1, value=6, step=1))
    context = int(st.number_input("Context length", min_value=8, value=64, step=8))
    gamma_energy = float(st.slider("Energy regularization γ", 0.0, 2.0, 0.5, 0.05))
    eta_work = float(st.slider("Work penalty η", 0.0, 2.0, 0.5, 0.05))
    lam_smooth = float(st.slider("Smoothness λ", 0.0, 2.0, 0.2, 0.05))

    st.markdown("**Future accuracy (optional)**")
    physicsplus_train = bool(
        st.toggle(
            "PhysicsPlus training (elastic + sticky)",
            value=bool(st.session_state.get("physicsplus_train", False)),
            help="Shapes token energies using springiness (stiff edges) + sticky contacts before training.",
        )
    )
    st.session_state["physicsplus_train"] = physicsplus_train
    if physicsplus_train:
        cutoff = float(st.slider("Contact cutoff (Å)", 4.0, 14.0, float(st.session_state.get("pp_cutoff", 8.0)), 0.5))
        sticky_q = float(st.slider("Sticky occupancy q", 0.50, 0.99, float(st.session_state.get("pp_sticky_q", 0.80)), 0.01))
        sticky_margin = float(st.slider("Sticky margin (Å)", 0.2, 3.0, float(st.session_state.get("pp_sticky_margin", 1.2)), 0.1))
        w_elastic_train = float(st.slider("Elastic weight", 0.0, 1.5, float(st.session_state.get("pp_w_elastic", 0.35)), 0.05))
        w_sticky_train = float(st.slider("Sticky weight", 0.0, 1.5, float(st.session_state.get("pp_w_sticky", 0.25)), 0.05))
        st.session_state.update(
            {
                "pp_cutoff": cutoff,
                "pp_sticky_q": sticky_q,
                "pp_sticky_margin": sticky_margin,
                "pp_w_elastic": w_elastic_train,
                "pp_w_sticky": w_sticky_train,
            }
        )
    else:
        cutoff = float(st.session_state.get("pp_cutoff", 8.0))
        sticky_q = float(st.session_state.get("pp_sticky_q", 0.80))
        sticky_margin = float(st.session_state.get("pp_sticky_margin", 1.2))
        w_elastic_train = float(st.session_state.get("pp_w_elastic", 0.35))
        w_sticky_train = float(st.session_state.get("pp_w_sticky", 0.25))

    st.divider()
    st.subheader("Prediction")

    horizon_unit_display = st.selectbox(
        "Prediction horizon unit",
        TIME_UNITS_DISPLAY,
        index=int(st.session_state.get("horizon_unit_idx", TIME_UNITS_DISPLAY.index(time_unit_display))),
        help="Convenience unit. Converted to the trajectory time unit for rollout.",
    )
    st.session_state["horizon_unit_idx"] = int(TIME_UNITS_DISPLAY.index(horizon_unit_display))
    horizon_val = float(st.number_input("Prediction horizon", min_value=0.0, value=float(st.session_state.get("horizon_val", 1.0)), step=1.0))
    st.session_state["horizon_val"] = float(horizon_val)

    # Convert the user-entered horizon to the trajectory unit used by dt/time annotations.
    horizon = float(_convert_time(horizon_val, horizon_unit_display, time_unit_display))
    st.caption(f"= {horizon:.6g} {time_unit_display}")

    beta = float(st.slider("Energy bias β", 0.0, 5.0, 1.0, 0.1))
    temperature = float(st.slider("Temperature", 0.0, 2.0, 0.8, 0.05))
    greedy = bool(st.checkbox("Greedy (argmax)", value=False))

    st.divider()
    st.subheader("Decode")
    decode_steps = int(st.number_input("Decode steps", min_value=50, value=200, step=50))
    r_min = float(st.number_input("Repulsion r_min", value=3.2, step=0.1))

    st.divider()
    with st.expander("Residue ring (data-driven)", expanded=False):
        st.caption("Drives the circular ring using residue–residue distance-change signals (|Δd|).")
        ring_nodes = int(st.number_input("Ring nodes (sampled residues)", min_value=8, value=int(st.session_state.get("ring_nodes", 64)), step=8))
        ring_knn = int(st.number_input("kNN edges per node", min_value=1, value=int(st.session_state.get("ring_knn", 3)), step=1))
        ring_cutoff = float(st.number_input("Contact cutoff (Å)", min_value=0.0, value=float(st.session_state.get("ring_cutoff", 8.0)), step=0.5))
        ring_max_edges = int(st.number_input("Max ring edges", min_value=20, value=int(st.session_state.get("ring_max_edges", 220)), step=20))
        ring_emit_decode = int(st.number_input("Ring update stride during decode", min_value=1, value=int(st.session_state.get("ring_emit_decode", 5)), step=1))
        ring_emit_movie = int(st.number_input("Ring update stride during movie frames", min_value=1, value=int(st.session_state.get("ring_emit_movie", 1)), step=1))

        st.session_state["ring_nodes"] = int(ring_nodes)
        st.session_state["ring_knn"] = int(ring_knn)
        st.session_state["ring_cutoff"] = float(ring_cutoff)
        st.session_state["ring_max_edges"] = int(ring_max_edges)
        st.session_state["ring_emit_decode"] = int(ring_emit_decode)
        st.session_state["ring_emit_movie"] = int(ring_emit_movie)

    st.divider()
    build_btn = st.button("Build + Train engine", type="primary", use_container_width=True)
    predict_btn = st.button("Predict pause structure", type="secondary", use_container_width=True)


# Re-inject CSS after sidebar toggles
inject_css(bool(st.session_state.get("reduced_motion", False)))


# -----------------------------
# Top: hero + pipeline + bus
# -----------------------------

hero()

# Use placeholders so the pipeline lights and bus can refresh live during long runs.
_PIPE_PH = st.empty()
_BUS_PH = st.empty()


def render_top() -> None:
    pipeline_bar(
        int(st.session_state["telemetry"].get("stage_idx", -1)),
        float(st.session_state["telemetry"].get("p", 0.0)),
        container=_PIPE_PH,
    )
    bus_render(container=_BUS_PH)


render_top()

# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs(["Dashboard", "Insights", "Activity", "Help", "Contact"])


def neon_wave_preview(y: List[float], *, stage_idx: int, title: str) -> go.Figure:
    y = (y or [])[:420]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(width=2), name=title))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.01, font=dict(size=14, color="rgba(238,240,255,0.92)")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        font=dict(color="rgba(238,240,255,0.86)"),
        showlegend=False,
    )
    return fig


# -----------------------------
# Dashboard
# -----------------------------

with tabs[0]:
    progress_cb = make_progress_cb(render_hook=render_top)

    # Top status strip (moved into Dashboard so tabs are the only navigation)
    status_row()

    st.subheader("Actions")
    st.markdown(
        """<div class='tdp-card-strong'>
        <div style='font-weight:900; font-size:18px;'>One pipeline, one bus</div>
        <div style='color:rgba(238,240,255,0.68)'>Build trains the engine. Predict rolls out a pause structure, decodes it, then proves it with RMSD.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    colA, colB = st.columns([1, 1])
    with colA:
        if build_btn:
            try:
                log("Build requested")
                _set_bus(0, 0.0, "preprocess", _carrier_wave(0.1), "Preprocess starting...")
                render_top()
                st.session_state["traj"] = build_traj(
                    demo=demo,
                    upload_mode=upload_mode,
                    top_upload=top_upload,
                    traj_upload=traj_upload,
                    top_url=top_url,
                    traj_url=traj_url,
                    selection=selection,
                    stride=stride,
                    time_total=time_total,
                    time_unit=time_unit,
                    k_nn=k_nn,
                    progress_cb=progress_cb,
                )
                st.session_state["engine"] = None
                st.session_state["last_pred"] = None
                log("Trajectory ready")

                ew = EnergyWeights(r_min=float(r_min))
                device = _device_string().split(" ")[0]
                cfg = TrainConfig(
                    context=int(context),
                    epochs=int(epochs),
                    gamma_energy=float(gamma_energy),
                    eta_work=float(eta_work),
                    lam_smooth=float(lam_smooth),
                    device=device,
                )

                # Optional: physics-plus pipeline integration (elastic + sticky priors)
                if bool(st.session_state.get("physicsplus_train", False)):
                    st.session_state["engine"] = build_engine_future(
                        st.session_state["traj"],
                        m=int(m),
                        K=int(K),
                        energy_w=ew,
                        train_cfg=cfg,
                        seed=int(st.session_state["seed"]),
                        cutoff=float(st.session_state.get("pp_cutoff", 8.0)),
                        sticky_q=float(st.session_state.get("pp_sticky_q", 0.80)),
                        sticky_margin=float(st.session_state.get("pp_sticky_margin", 1.2)),
                        w_elastic=float(st.session_state.get("pp_w_elastic", 0.35)),
                        w_sticky=float(st.session_state.get("pp_w_sticky", 0.25)),
                        progress_cb=progress_cb,
                    )
                else:
                    st.session_state["engine"] = build_engine(
                        st.session_state["traj"],
                        m=int(m),
                        K=int(K),
                        energy_w=ew,
                        train_cfg=cfg,
                        seed=int(st.session_state["seed"]),
                        progress_cb=progress_cb,
                    )
                log("Engine trained")
                _set_bus(2, 1.0, "train", st.session_state["bus_series"].get("loss", [])[-240:], "Engine ready.")
                render_top()
                st.success("Engine trained.")
            except Exception as e:
                log(f"Build failed: {e}", kind="error")
                st.error(str(e))

    with colB:
        if predict_btn:
            if st.session_state.get("engine") is None:
                st.warning("Train an engine first.")
            else:
                try:
                    log("Prediction requested")
                    _set_bus(3, 0.0, "predict", _carrier_wave(0.15), "Predict starting...")
                    render_top()
                    rcfg = RolloutConfig(beta_energy=float(beta), temperature=float(temperature), greedy=bool(greedy))
                    dw = DecodeWeights(r_min=float(r_min))
                    device = _device_string().split(" ")[0]
                    ring_spec = build_ring_spec(st.session_state["engine"], n_nodes=int(st.session_state.get("ring_nodes", 64)), knn=int(st.session_state.get("ring_knn", 3)), cutoff=float(st.session_state.get("ring_cutoff", 8.0)), max_edges=int(st.session_state.get("ring_max_edges", 220)))
                    st.session_state["last_pred"] = predict_pause_structures(
                        st.session_state["engine"],
                        float(horizon),
                        rcfg,
                        decode_w=dw,
                        decode_steps=int(decode_steps),
                        device=device,
                        progress_cb=progress_cb,
                        ring_spec=ring_spec,
                        ring_emit_every=int(st.session_state.get("ring_emit_decode", 5)),
                    )

                    # Proof: RMSD vs last frame (robust to rare decode issues)
                    X_last = st.session_state["engine"].traj.X[-1]
                    X_pred = st.session_state["last_pred"].get("X_rec")
                    rmsd = float("nan")
                    disp = None
                    try:
                        if X_pred is not None:
                            Xp = np.asarray(X_pred, dtype=np.float32)
                            if np.isfinite(Xp).all():
                                rmsd = float(rmsd_kabsch(X_last, Xp))
                                disp = per_site_displacement(X_last, Xp, align=True).astype(np.float32)
                            else:
                                log("Warning: predicted coordinates contained non-finite values; RMSD skipped", kind="warn")
                    except Exception as _e:
                        log(f"Warning: RMSD computation failed: {_e}", kind="warn")

                    st.session_state["last_pred"]["rmsd"] = float(rmsd)
                    if disp is not None:
                        st.session_state["last_pred"]["disp"] = disp

                    if disp is not None:
                        st.session_state["bus_series"]["disp"] = disp.tolist()[-420:]
                        _set_bus(5, 1.0, "proof", disp.tolist(), f"Proof: RMSD(last→pred) = {rmsd:.3f} Å")
                    else:
                        _set_bus(5, 1.0, "proof", _carrier_wave(0.15), f"Proof: RMSD unavailable (non-finite decode)")
                    render_top()
                    render_top()
                    log(f"Prediction complete (RMSD={rmsd:.3f})", kind="success")
                    st.success("Prediction complete.")
                except Exception as e:
                    log(f"Predict failed: {e}", kind="error")
                    st.error(str(e))

    st.divider()

    traj: Optional[TrajectoryData] = st.session_state.get("traj")
    engine = st.session_state.get("engine")
    out = st.session_state.get("last_pred")

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.subheader("Preprocess preview")
        if traj is None:
            st.caption("No trajectory yet.")
        else:
            st.write(
                f"Frames: **{traj.X.shape[0]}** | Sites: **{traj.X.shape[1]}** | Edges: **{traj.edges.n_edges}** | dt: **{traj.dt:.4g} {traj.time_unit}**"
            )
            pchart(neon_wave_preview(traj.d[: min(len(traj.d), 420), 0].tolist(), stage_idx=0, title="Edge[0] distance"), use_container_width=True)

    with c2:
        st.subheader("Training + energy")
        if engine is None:
            st.caption("Train the engine to see training curves.")
        else:
            hist = engine.train_report.history
            y = [float(h["loss_ce"]) for h in hist]
            pchart(neon_wave_preview(y, stage_idx=2, title="CE loss (by epoch)"), use_container_width=True)

            E_t = compute_frame_energies(engine.traj, engine.stats, EnergyWeights(r_min=float(r_min)))
            pchart(neon_wave_preview(E_t[: min(len(E_t), 420)].tolist(), stage_idx=2, title="Surrogate energy (per frame)"), use_container_width=True)

    st.divider()

    st.subheader("Triplet waves → tokens")
    if engine is None:
        st.info("Train first.")
    else:
        tri = triplet_from_latent(engine.tok.z)
        tA, tB = st.columns([1, 1], gap="large")
        with tA:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=tri.plus[:420], mode="lines", name="|+⟩", line=dict(width=2, color="rgba(255,85,115,0.92)")))
            fig.add_trace(go.Scatter(y=tri.zero[:420], mode="lines", name="|0⟩", line=dict(width=2, color="rgba(255,119,199,0.85)")))
            fig.add_trace(go.Scatter(y=tri.minus[:420], mode="lines", name="|−⟩", line=dict(width=2, color="rgba(137,255,171,0.85)")))
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=34, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Triplet waves", x=0.01, font=dict(size=14, color="rgba(238,240,255,0.92)")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                font=dict(color="rgba(238,240,255,0.86)"),
            )
            pchart(fig, use_container_width=True)

        with tB:
            tok_tail = engine.tok.tokens[-420:]
            fig = go.Figure(go.Heatmap(z=[tok_tail], colorscale="Turbo", showscale=False))
            fig.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=34, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Token strip (latest)", x=0.01, font=dict(size=14, color="rgba(238,240,255,0.92)")),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            pchart(fig, use_container_width=True)

    st.divider()

    st.subheader("Prediction → decode → proof")
    if engine is None or out is None:
        st.caption("Run a prediction to populate this section.")
    else:
        rmsd = float(out.get("rmsd", float("nan")))
        st.metric("RMSD (last frame vs predicted pause)", f"{rmsd:.3f} Å")

        # Export PDB
        tmp_dir = Path(tempfile.gettempdir())
        out_pdb = tmp_dir / "pause_structure_ca.pdb"
        export_pause_pdb(engine, out["X_rec"], int(out["token_last"]), str(out_pdb))
        pdb_bytes = out_pdb.read_bytes()

        dA, dB = st.columns([1, 1], gap="large")
        with dA:
            ent = out["rollout"]["entropies"].astype(np.float32).tolist()
            pchart(neon_wave_preview(ent, stage_idx=3, title="Rollout entropy"), use_container_width=True)
            st.download_button(
                "Download pause structure (CA-only PDB)",
                data=pdb_bytes,
                file_name="pause_structure_ca.pdb",
                mime="chemical/x-pdb",
                use_container_width=True,
            )

        with dB:
            if bool(st.session_state.get("enable_3d", True)):
                try:
                    render_ngl(pdb_bytes, height=380)
                except Exception:
                    st.info("3D viewer unavailable in this environment. Download the PDB instead.")
            else:
                st.caption("3D viewer is off. Toggle it in the sidebar.")

        # Export a mini report (JSON + PDB + activity)
        report = {
            "config": {
                "seed": st.session_state.get("seed"),
                "selection": selection,
                "stride": stride,
                "time_total": time_total,
                "time_unit": time_unit,
                "m": m,
                "K": K,
                "k_nn": k_nn,
                "epochs": epochs,
                "context": context,
                "gamma_energy": gamma_energy,
                "eta_work": eta_work,
                "lam_smooth": lam_smooth,
                "horizon": horizon,
                "beta": beta,
                "temperature": temperature,
                "greedy": greedy,
                "decode_steps": decode_steps,
                "r_min": r_min,
            },
            "metrics": {"rmsd": rmsd},
            "activity": st.session_state.get("activity", [])[-500:],
        }
        buf = io.BytesIO()
        import zipfile

        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pause_structure_ca.pdb", pdb_bytes)
            zf.writestr("report.json", json.dumps(report, indent=2))
        buf.seek(0)
        st.download_button(
            "Download run report (zip)",
            data=buf.read(),
            file_name="tdphysics_run_report.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.divider()

        # -----------------------------
        # Future movie export (beam + physics-plus decode)
        # -----------------------------
        with st.expander("Future movie export: beam rollout + multi-frame decode (multi-model PDB)", expanded=False):
            st.caption(
                "This exports a playable multi-model PDB (MODEL/ENDMDL blocks). "
                "Use it in PyMOL/VMD as a trajectory."
            )

            mA, mB, mC = st.columns([1, 1, 1], gap="large")
            with mA:
                n_frames = int(st.slider("Frames", 5, 60, int(st.session_state.get("movie_frames", 20)), 1))
                decode_steps_movie = int(
                    st.slider("Decode steps per frame", 60, 400, int(st.session_state.get("movie_decode_steps", 140)), 10)
                )
                decode_physics = bool(
                    st.toggle(
                        "PhysicsPlus decode (elastic + sticky)",
                        value=bool(st.session_state.get("movie_decode_physics", True)),
                    )
                )
            with mB:
                use_beam = bool(st.toggle("Beam search rollout", value=bool(st.session_state.get("movie_beam", True))))
                beam_width = int(st.slider("Beam width", 2, 24, int(st.session_state.get("movie_beam_width", 8)), 1))
                topk = int(st.slider("Top-k expansions", 8, 64, int(st.session_state.get("movie_topk", 24)), 1))
            with mC:
                work_penalty = float(
                    st.slider("Beam work penalty", 0.0, 2.0, float(st.session_state.get("movie_work_pen", 0.50)), 0.05)
                )
                beta_m = float(st.slider("Energy bias β (movie)", 0.0, 5.0, float(st.session_state.get("movie_beta", beta)), 0.1))
                temp_m = float(
                    st.slider("Temperature (movie)", 0.0, 2.0, float(st.session_state.get("movie_temp", temperature)), 0.05)
                )

            st.session_state.update(
                {
                    "movie_frames": n_frames,
                    "movie_decode_steps": decode_steps_movie,
                    "movie_decode_physics": decode_physics,
                    "movie_beam": use_beam,
                    "movie_beam_width": beam_width,
                    "movie_topk": topk,
                    "movie_work_pen": work_penalty,
                    "movie_beta": beta_m,
                    "movie_temp": temp_m,
                }
            )

            run_movie = st.button("Predict + build movie PDB", use_container_width=True)
            if run_movie:
                movie_status = st.empty()
                movie_bar = st.progress(0)
                movie_status.info("Starting movie generation...")

                # Wrap the global progress callback so we also drive the local bar + status.
                _base_cb = progress_cb

                def movie_progress_cb(event: str, payload: Optional[dict] = None) -> None:
                    payload = payload or {}
                    ev = str(event)
                    if ev.startswith("pipeline."):
                        ev = ev[len("pipeline.") :]

                    if ev.startswith("movie"):
                        p_loc = float(payload.get("p", 0.0))
                        # prefer frame-based status if available
                        if ev.endswith("start"):
                            nf = int(payload.get("n_frames", 0) or 0)
                            movie_status.info(f"Generating movie... (0/{nf} frames decoded)")
                        elif ev.endswith("frame"):
                            fr = int(payload.get("frame", 0) or 0)
                            nf = int(payload.get("n_frames", 0) or 0)
                            movie_status.info(f"Generating movie... (frame {fr}/{nf})")
                        elif ev.endswith("done"):
                            movie_status.success("Movie generated ✔")
                        movie_bar.progress(int(max(0.0, min(1.0, p_loc)) * 100))

                    # Always forward to the global bus callback
                    try:
                        _base_cb(event, payload)
                    except Exception:
                        pass

                try:
                    log("Movie export requested")
                    bcfg = BeamConfig(beam_width=int(beam_width), topk=int(topk), work_penalty=float(work_penalty))
                    device = _device_string().split(" ")[0]
                    ring_spec = build_ring_spec(engine, n_nodes=int(st.session_state.get("ring_nodes", 64)), knn=int(st.session_state.get("ring_knn", 3)), cutoff=float(st.session_state.get("ring_cutoff", 8.0)), max_edges=int(st.session_state.get("ring_max_edges", 220)))
                    movie = predict_future_movie(
                        engine,
                        horizon_time=float(horizon),
                        n_frames=int(n_frames),
                        use_beam=bool(use_beam),
                        beam_cfg=bcfg,
                        beta_energy=float(beta_m),
                        temperature=float(temp_m),
                        greedy=False,
                        decode_w=DecodeWeights(r_min=float(r_min)),
                        decode_steps=int(decode_steps_movie),
                        decode_physics=bool(decode_physics),
                        priors=getattr(engine, "physics_priors", None),
                        device=device,
                        progress_cb=movie_progress_cb,
                        ring_spec=ring_spec,
                        ring_emit_every=int(st.session_state.get("ring_emit_movie", 1)),
                    )
                    st.session_state["last_movie"] = movie

                    movie_status.info("Writing multi-model PDB...")
                    movie_bar.progress(95)

                    tmp_dir = Path(tempfile.gettempdir())
                    out_movie_pdb = tmp_dir / "future_movie_ca.pdb"
                    export_future_movie_pdb(engine, movie, str(out_movie_pdb))
                    movie_bytes = out_movie_pdb.read_bytes()

                    # Persist movie bytes so the download + plots survive reruns.
                    st.session_state["last_movie_pdb_bytes"] = movie_bytes

                    movie_status.success("Movie ready for download ✔")
                    movie_bar.progress(100)

                    # quick preview plots
                    ent = movie["rollout"]["entropies"].astype(np.float32).tolist()
                    pchart(neon_wave_preview(ent, stage_idx=3, title="Movie rollout entropy"), use_container_width=True)

                    st.download_button(
                        "Download future movie (CA-only multi-model PDB)",
                        data=movie_bytes,
                        file_name="future_movie_ca.pdb",
                        mime="chemical/x-pdb",
                        use_container_width=True,
                    )
                    st.success("Movie PDB ready.")
                except Exception as e:
                    log(f"Movie export failed: {e}", kind="error")
                    try:
                        movie_status.error(f"Movie export failed: {e}")
                        movie_bar.progress(0)
                    except Exception:
                        pass
                    st.error(str(e))

            # Always show the last movie (if available), even after rerun.
            if st.session_state.get("last_movie") is not None:
                try:
                    movie = st.session_state.get("last_movie")
                    ent = movie["rollout"]["entropies"].astype(np.float32).tolist()
                    if len(ent) > 0:
                        pchart(neon_wave_preview(ent, stage_idx=3, title="Movie rollout entropy (last run)"), use_container_width=True)

                    movie_bytes = st.session_state.get("last_movie_pdb_bytes")
                    if movie_bytes is None:
                        tmp_dir = Path(tempfile.gettempdir())
                        out_movie_pdb = tmp_dir / "future_movie_ca.pdb"
                        export_future_movie_pdb(engine, movie, str(out_movie_pdb))
                        movie_bytes = out_movie_pdb.read_bytes()
                        st.session_state["last_movie_pdb_bytes"] = movie_bytes

                    st.download_button(
                        "Download last movie (CA-only multi-model PDB)",
                        data=movie_bytes,
                        file_name="future_movie_ca.pdb",
                        mime="chemical/x-pdb",
                        use_container_width=True,
                    )
                except Exception:
                    st.info("Movie exists in memory, but preview/export failed in this environment. Try regenerating.")


# -----------------------------
# Insights
# -----------------------------

with tabs[1]:
    st.subheader("Insights")
    engine = st.session_state.get("engine")
    if engine is None:
        st.caption("Train the engine to unlock insights.")
    else:
        z = engine.tok.z
        c1, c2 = st.columns([1.2, 0.8], gap="large")
        with c1:
            if z.shape[1] >= 2:
                fig = go.Figure(go.Scattergl(x=z[:4000, 0], y=z[:4000, 1], mode="markers", marker=dict(size=3, opacity=0.65)))
                fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=36, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=dict(text="Latent scatter (z0 vs z1)", x=0.01, font=dict(size=14, color="rgba(238,240,255,0.92)")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="z[0]"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="z[1]"),
                    font=dict(color="rgba(238,240,255,0.86)"),
                )
                pchart(fig, use_container_width=True)
            else:
                st.info("Latent dimension < 2, scatter unavailable.")

        with c2:
            toks = engine.tok.tokens
            counts = np.bincount(toks, minlength=int(engine.train_report.vocab_size)).astype(np.float32)
            top = np.argsort(counts)[::-1][:24]
            fig2 = go.Figure(go.Bar(x=top, y=counts[top]))
            fig2.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=36, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Top token usage", x=0.01, font=dict(size=14, color="rgba(238,240,255,0.92)")),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="count"),
                font=dict(color="rgba(238,240,255,0.86)"),
            )
            pchart(fig2, use_container_width=True)


# -----------------------------
# Activity
# -----------------------------

with tabs[2]:
    st.subheader("Activity log")
    activity = st.session_state.get("activity", [])
    if not activity:
        st.caption("No events yet. Build/train or predict to generate activity.")
    else:
        kinds = sorted({a.get("kind", "info") for a in activity})
        ksel = st.multiselect("Filter", options=kinds, default=kinds)
        view = [a for a in activity if a.get("kind", "info") in set(ksel)]
        lines = []
        for a in view[-400:]:
            lines.append(f"[{a.get('t','--:--:--')}] {a.get('kind','info').upper():7s}  {a.get('msg','')}")
        st.code("\n".join(lines), language="text")


# -----------------------------
# Help
# -----------------------------

with tabs[3]:
    st.subheader("Usage")
    st.markdown(
        """
**Quick start**

1. **Load data** (sidebar)
   - Upload a **Topology** (PDB/GRO/PSF/TPR).
   - Upload a **Trajectory** (XTC/DCD/TRR/NC) or leave blank for topology-only.
   - Optional: switch **Input mode = URL** to stream large files from a link.

2. **Set timing**
   - Choose **Trajectory time unit**.
   - Set **Total trajectory time** (or leave **0** to infer dt from the trajectory when available).

3. **Build + Train engine**
   - Click **Build + Train engine**.
   - Watch the pipeline cards light up as stages finish.

4. **Predict**
   - Set **Prediction horizon** (value + unit).
   - Click **Predict pause structure**.
   - Check the **Proof** (RMSD + displacement) and export the pause PDB.

5. **Movie export**
   - Open the **Future movie export** expander.
   - Click **Predict + build movie PDB**.
   - A progress bar shows frame generation, then you can download the multi-model PDB.
"""
    )

    with st.expander("Tips", expanded=False):
        st.markdown(
            """
- For long horizons, prefer fewer movie frames (coarser cadence) and enable **Beam search rollout**.
- If your trajectory is CA-only, set selection to `name CA`.
- If you see memory pressure, use **Stride** and prefer URL mode for huge files.
"""
        )


# -----------------------------
# Contact
# -----------------------------

with tabs[4]:
    st.subheader("Contact")
    st.markdown("Support and collaborations:")
    st.code("bessuman.academia@gmail.com\nwww.proteinsimulationconsulting.com", language="text")
    st.caption("Include your topology/trajectory formats, approximate residue count, and your target horizon when reaching out.")
