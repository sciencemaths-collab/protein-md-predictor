from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .utils import hsv_from_latent, hsv_to_rgb

def plot_token_series(tokens: np.ndarray, title: str = "Tokens over time") -> plt.Figure:
    fig = plt.figure()
    plt.plot(tokens, linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Token")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_energy_series(E_t: np.ndarray, title: str = "Surrogate energy over time") -> plt.Figure:
    fig = plt.figure()
    plt.plot(E_t, linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Energy (a.u.)")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_triplet(plus: np.ndarray, zero: np.ndarray, minus: np.ndarray, title: str = "Triplet decomposition") -> plt.Figure:
    fig = plt.figure()
    plt.plot(plus, label="|+>", linewidth=1.0)
    plt.plot(zero, label="|0>", linewidth=1.0)
    plt.plot(minus, label="|->", linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig

def latent_color_strip(z: np.ndarray, coherence: np.ndarray, title: str = "Latent→Color strip") -> plt.Figure:
    z2 = z[:, :2]
    mag = np.linalg.norm(z, axis=-1)
    hsv = hsv_from_latent(z2=z2, mag=mag, conf=coherence)
    rgb = hsv_to_rgb(hsv)
    fig = plt.figure(figsize=(10, 1.4))
    plt.imshow(rgb[None, :, :], aspect="auto")
    plt.yticks([])
    plt.xticks([])
    plt.title(title)
    plt.tight_layout()
    return fig
