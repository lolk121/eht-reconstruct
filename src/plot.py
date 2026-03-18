"""
UV-coverage and data exploration plots.
 
Usage:
    from src.plot import plot_uv_coverage, plot_amplitude_vs_baseline
    from src.parse import load_uvfits
 
    obs = load_uvfits("data/M87_HI_hops_netcal.uvfits")
    plot_uv_coverage(obs, save="data/uv_coverage.png")
    plot_amplitude_vs_baseline(obs, save="data/amp_vs_uv.png")
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
 
from src.parse import Observation
 
 
def plot_uv_coverage(
    obs: Observation,
    color_by_baseline: bool = True,
    save: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the uv-coverage. Shows both (u,v) and conjugate (-u,-v)
    since visibilities are Hermitian for a real sky brightness.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
 
    if color_by_baseline:
        # Unique color per baseline pair
        bl_ids = obs.ant1 * 100 + obs.ant2
        unique_bls = np.unique(bl_ids)
        cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_bls)))
 
        for i, bl in enumerate(unique_bls):
            mask = bl_ids == bl
            a1, a2 = obs.ant1[mask][0], obs.ant2[mask][0]
            label = f"{obs.stations[a1]}-{obs.stations[a2]}"
            u_gl = obs.u_glambda[mask]
            v_gl = obs.v_glambda[mask]
            ax.scatter(u_gl, v_gl, s=1, color=cmap[i], label=label)
            ax.scatter(-u_gl, -v_gl, s=1, color=cmap[i])  # conjugate
    else:
        ax.scatter(obs.u_glambda, obs.v_glambda, s=1, c="steelblue", alpha=0.5)
        ax.scatter(-obs.u_glambda, -obs.v_glambda, s=1, c="steelblue", alpha=0.5)
 
    ax.set_xlabel("u (Gλ)")
    ax.set_ylabel("v (Gλ)")
    ax.set_title(f"UV Coverage — {obs.source} ({obs.freq_hz/1e9:.0f} GHz)")
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", lw=0.5, alpha=0.3)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.3)
 
    if color_by_baseline and len(obs.baselines) <= 28:
        ax.legend(fontsize=6, markerscale=5, loc="upper right", ncol=2)
 
    fig.tight_layout()
 
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
 
    return fig
 
 
def plot_amplitude_vs_uvdist(
    obs: Observation,
    save: str | Path | None = None,
) -> plt.Figure:
    """
    Visibility amplitude vs uv-distance.
    Should show a characteristic dip/ring structure for M87*.
    """
    uvdist = np.hypot(obs.u_glambda, obs.v_glambda)
    amp = np.abs(obs.vis)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(uvdist, amp, s=2, alpha=0.4, c="steelblue")
    ax.set_xlabel("uv-distance (Gλ)")
    ax.set_ylabel("Visibility Amplitude (Jy)")
    ax.set_title(f"Amplitude vs UV-Distance — {obs.source}")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
 
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
 
    return fig
 
 
def plot_phase_vs_uvdist(
    obs: Observation,
    save: str | Path | None = None,
) -> plt.Figure:
    """Visibility phase vs uv-distance."""
    uvdist = np.hypot(obs.u_glambda, obs.v_glambda)
    phase = np.angle(obs.vis, deg=True)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(uvdist, phase, s=2, alpha=0.3, c="coral")
    ax.set_xlabel("uv-distance (Gλ)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title(f"Phase vs UV-Distance — {obs.source}")
    ax.set_ylim(-180, 180)
    fig.tight_layout()
 
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
 
    return fig
 
 
def plot_amplitude_vs_time(
    obs: Observation,
    save: str | Path | None = None,
) -> plt.Figure:
    """Visibility amplitude over time, colored by baseline."""
    fig, ax = plt.subplots(figsize=(12, 5))
 
    bl_ids = obs.ant1 * 100 + obs.ant2
    unique_bls = np.unique(bl_ids)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_bls)))
 
    # Convert MJD to hours from start
    t_hours = (obs.time_mjd - obs.time_mjd.min()) * 24.0
 
    for i, bl in enumerate(unique_bls):
        mask = bl_ids == bl
        a1, a2 = obs.ant1[mask][0], obs.ant2[mask][0]
        label = f"{obs.stations[a1]}-{obs.stations[a2]}"
        ax.scatter(t_hours[mask], np.abs(obs.vis[mask]), s=3, color=cmap[i], label=label, alpha=0.6)
 
    ax.set_xlabel("Time (hours from start)")
    ax.set_ylabel("Amplitude (Jy)")
    ax.set_title(f"Amplitude vs Time — {obs.source}")
 
    if len(unique_bls) <= 28:
        ax.legend(fontsize=5, markerscale=3, loc="upper right", ncol=3)
 
    fig.tight_layout()
 
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
 
    return fig
 
