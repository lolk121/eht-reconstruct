"""
Phase 2: Dirty image reconstruction.

Grids irregularly sampled visibilities onto a regular Fourier grid,
then inverse FFTs to produce the dirty image (first crude reconstruction)
and dirty beam (point spread function).

The dirty image = true sky convolved with the dirty beam.
It's artefacted but proves the pipeline works and gives you your
first look at M87*.

Usage:
    from src.parse import load_uvfits
    from src.dirty import make_dirty_image, plot_dirty

    obs = load_uvfits("data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits")
    dirty, beam, extent = make_dirty_image(obs, npix=256, fov_uas=200.0)
    plot_dirty(dirty, beam, extent, save="data/dirty_image.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.parse import Observation


# Microarcseconds to radians
UAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1e6)


def make_dirty_image(
    obs: Observation,
    npix: int = 256,
    fov_uas: float = 200.0,
    weighting: str = "uniform",
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    """
    Compute the dirty image and dirty beam via gridding + inverse FFT.

    Parameters
    ----------
    obs : Observation
        Parsed visibility data.
    npix : int
        Output image size (npix x npix).
    fov_uas : float
        Field of view in microarcseconds.
    weighting : str
        'uniform' — equal weight per grid cell (higher resolution, more noise).
        'natural' — weight by sampling density (lower resolution, less noise).

    Returns
    -------
    dirty_image : array, shape (npix, npix)
    dirty_beam : array, shape (npix, npix)
    extent : [left, right, bottom, top] in microarcseconds (for imshow)
    """
    fov_rad = fov_uas * UAS_TO_RAD
    pixel_rad = fov_rad / npix

    # uv-cell size: 1 / (npix * pixel_size)
    du = 1.0 / (npix * pixel_rad)

    # Map (u, v) to grid indices
    # Centre of grid is (npix//2, npix//2)
    u_idx = np.round(obs.u / du).astype(np.int64) + npix // 2
    v_idx = np.round(obs.v / du).astype(np.int64) + npix // 2

    # Also include conjugate visibilities: V(-u,-v) = conj(V(u,v))
    u_idx_all = np.concatenate([u_idx, -u_idx + npix])
    v_idx_all = np.concatenate([v_idx, -v_idx + npix])
    vis_all = np.concatenate([obs.vis, np.conj(obs.vis)])
    weights_all = np.concatenate([1.0 / obs.sigma**2, 1.0 / obs.sigma**2])

    # Keep only points that land inside the grid
    valid = (
        (u_idx_all >= 0) & (u_idx_all < npix) &
        (v_idx_all >= 0) & (v_idx_all < npix)
    )
    u_idx_all = u_idx_all[valid]
    v_idx_all = v_idx_all[valid]
    vis_all = vis_all[valid]
    weights_all = weights_all[valid]

    # Grid the visibilities
    vis_grid = np.zeros((npix, npix), dtype=np.complex128)
    weight_grid = np.zeros((npix, npix), dtype=np.float64)
    sampling_grid = np.zeros((npix, npix), dtype=np.complex128)

    if weighting == "natural":
        # Natural weighting: each visibility contributes its own weight
        for i in range(len(vis_all)):
            ui, vi = u_idx_all[i], v_idx_all[i]
            w = weights_all[i]
            vis_grid[vi, ui] += vis_all[i] * w
            weight_grid[vi, ui] += w
            sampling_grid[vi, ui] += w
    else:
        # Uniform weighting: all cells get equal weight
        for i in range(len(vis_all)):
            ui, vi = u_idx_all[i], v_idx_all[i]
            vis_grid[vi, ui] += vis_all[i]
            weight_grid[vi, ui] += 1.0
            sampling_grid[vi, ui] += 1.0

    # Normalise: average visibilities in each cell
    occupied = weight_grid > 0
    vis_grid[occupied] /= weight_grid[occupied]
    sampling_grid[occupied] = 1.0  # For beam: just mark as sampled

    # Inverse FFT → image
    dirty_image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid))))
    dirty_beam = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_grid))))

    # Normalise beam to peak = 1
    beam_max = dirty_beam.max()
    if beam_max > 0:
        dirty_beam /= beam_max
        dirty_image /= beam_max

    # Image extent in microarcseconds (for matplotlib imshow)
    half_fov = fov_uas / 2.0
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    return dirty_image, dirty_beam, extent


def plot_dirty(
    dirty_image: NDArray[np.float64],
    dirty_beam: NDArray[np.float64],
    extent: list[float],
    save: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the dirty image and dirty beam side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Dirty image
    vmax = np.percentile(dirty_image, 99.5)
    vmin = np.percentile(dirty_image, 0.5)
    im1 = ax1.imshow(
        dirty_image, origin="lower", extent=extent,
        cmap="afmhot", vmin=vmin, vmax=vmax,
    )
    ax1.set_xlabel("Relative RA (μas)")
    ax1.set_ylabel("Relative Dec (μas)")
    ax1.set_title("Dirty Image — M87*")
    ax1.invert_xaxis()  # RA increases to the left
    fig.colorbar(im1, ax=ax1, label="Flux density (Jy/beam)", shrink=0.8)

    # Dirty beam
    im2 = ax2.imshow(
        dirty_beam, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-0.3, vmax=1.0,
    )
    ax2.set_xlabel("Relative RA (μas)")
    ax2.set_ylabel("Relative Dec (μas)")
    ax2.set_title("Dirty Beam (PSF)")
    ax2.invert_xaxis()
    fig.colorbar(im2, ax=ax2, label="Response", shrink=0.8)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")

    return fig
