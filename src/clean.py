"""
Phase 3: CLEAN image reconstruction (Högbom 1974).

Iteratively deconvolves the dirty beam from the dirty image to reveal
the true source structure underneath the sidelobes.

Algorithm:
    1. Find peak in residual image
    2. Subtract gain * peak * shifted dirty beam
    3. Add gain * peak to component model
    4. Repeat until threshold or max iterations
    5. Fit a clean beam (Gaussian) to the dirty beam main lobe
    6. Final image = convolve(components, clean_beam) + residuals

Usage:
    from src.parse import load_uvfits
    from src.dirty import make_dirty_image
    from src.clean import hogbom_clean, plot_clean

    obs = load_uvfits("data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits")
    dirty, beam, extent = make_dirty_image(obs, npix=256, fov_uas=200.0)
    result = hogbom_clean(dirty, beam, n_iter=5000, gain=0.1, threshold=0.0005)
    plot_clean(result, extent, save="data/clean_image.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift
from scipy.optimize import curve_fit


@dataclass
class CleanResult:
    """Output of a CLEAN run."""

    restored: NDArray[np.float64]     # final image (model * clean_beam + residuals)
    components: NDArray[np.float64]   # accumulated point source model
    residuals: NDArray[np.float64]    # final residual image
    clean_beam: NDArray[np.float64]   # fitted Gaussian beam
    peak_history: list[float]         # peak residual at each iteration
    n_iter: int                       # iterations actually performed


def _fit_clean_beam(dirty_beam: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Fit a 2D Gaussian to the main lobe of the dirty beam.

    The clean beam replaces the dirty beam's messy sidelobes with a
    smooth Gaussian of the same resolution, giving the final image
    a well-defined point spread function.
    """
    npix = dirty_beam.shape[0]
    cy, cx = npix // 2, npix // 2

    # Extract a small patch around the central peak for fitting
    hw = max(npix // 8, 10)
    patch = dirty_beam[cy - hw:cy + hw, cy - hw:cy + hw]

    # Set up coordinate grids relative to patch centre
    y, x = np.mgrid[-hw:hw, -hw:hw].astype(np.float64)

    def gaussian_2d(coords, amp, sigma_x, sigma_y, theta):
        xr = coords[0]
        yr = coords[1]
        a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
        b = np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
        c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
        return amp * np.exp(-(a * xr**2 + 2 * b * xr * yr + c * yr**2))

    coords = np.vstack([x.ravel(), y.ravel()])
    try:
        popt, _ = curve_fit(
            gaussian_2d, coords, patch.ravel(),
            p0=[1.0, 3.0, 3.0, 0.0],
            bounds=([0.5, 0.5, 0.5, -np.pi], [1.5, hw, hw, np.pi]),
            maxfev=5000,
        )
    except RuntimeError:
        # Fallback: use a circular Gaussian with FWHM estimated from beam
        half_max = dirty_beam[cy, cx] / 2.0
        row = dirty_beam[cy, :]
        above = np.where(row >= half_max)[0]
        fwhm = above[-1] - above[0] if len(above) > 1 else 5
        sigma = fwhm / 2.355
        popt = [1.0, sigma, sigma, 0.0]

    # Generate full-size clean beam
    y_full, x_full = np.mgrid[-cy:npix - cy, -cx:npix - cx].astype(np.float64)
    coords_full = np.vstack([x_full.ravel(), y_full.ravel()])
    clean_beam = gaussian_2d(coords_full, *popt).reshape(npix, npix)

    # Normalise to peak = 1
    clean_beam /= clean_beam.max()

    return clean_beam


def hogbom_clean(
    dirty_image: NDArray[np.float64],
    dirty_beam: NDArray[np.float64],
    n_iter: int = 3000,
    gain: float = 0.05,
    threshold: float = 0.0001,
    window_radius: int | None = None,
) -> CleanResult:
    """
    Högbom CLEAN — iterative point-source subtraction.

    Parameters
    ----------
    dirty_image : array, shape (npix, npix)
    dirty_beam : array, shape (npix, npix)
    n_iter : int
        Maximum iterations.
    gain : float
        Loop gain — fraction of peak subtracted per iteration.
    threshold : float
        Stop when peak residual drops below this.
    window_radius : int, optional
        Only search for peaks within this radius (pixels) of image centre.
        Prevents CLEAN from chasing sidelobes at image edges.
        Defaults to npix // 4.

    Returns
    -------
    CleanResult
    """
    npix = dirty_image.shape[0]
    cy, cx = npix // 2, npix // 2

    # CLEAN window — circular mask around the centre
    if window_radius is None:
        window_radius = npix // 4
    yy, xx = np.ogrid[:npix, :npix]
    window = ((yy - cy)**2 + (xx - cx)**2) <= window_radius**2

    residuals = dirty_image.copy()
    components = np.zeros_like(dirty_image)
    peak_history = []
    min_peak = np.inf

    print(f"Running CLEAN: max {n_iter} iters, gain={gain}, threshold={threshold}, window_r={window_radius}px")

    for i in range(n_iter):
        # Find peak only within the CLEAN window
        masked_residuals = np.where(window, np.abs(residuals), 0.0)
        peak_idx = np.unravel_index(np.argmax(masked_residuals), residuals.shape)
        peak_val = residuals[peak_idx]
        abs_peak = abs(peak_val)

        peak_history.append(abs_peak)

        # Stop if below threshold
        if abs_peak < threshold:
            print(f"  Converged at iteration {i}: peak residual {abs_peak:.6f} < {threshold}")
            break

        # Stop if residuals are increasing (divergence detection)
        if abs_peak < min_peak:
            min_peak = abs_peak
        elif abs_peak > min_peak * 1.5 and i > 100:
            print(f"  Stopping at iteration {i}: residuals diverging ({abs_peak:.6f} > {min_peak:.6f} * 1.5)")
            break

        # Shift dirty beam to peak location and subtract
        dy = peak_idx[0] - cy
        dx = peak_idx[1] - cx
        shifted_beam = shift(dirty_beam, [dy, dx], order=1, mode="constant", cval=0.0)

        residuals -= gain * peak_val * shifted_beam
        components[peak_idx] += gain * peak_val

        if (i + 1) % 500 == 0:
            print(f"  Iteration {i + 1}: peak residual = {abs_peak:.6f}")

    else:
        print(f"  Reached max iterations ({n_iter}): peak residual = {peak_history[-1]:.6f}")

    # Fit clean beam to dirty beam main lobe
    print("Fitting clean beam...")
    clean_beam = _fit_clean_beam(dirty_beam)

    # Restore: convolve component model with clean beam, add residuals
    components_ft = np.fft.fft2(components)
    clean_beam_ft = np.fft.fft2(np.fft.ifftshift(clean_beam))
    restored = np.real(np.fft.ifft2(components_ft * clean_beam_ft)) + residuals

    return CleanResult(
        restored=restored,
        components=components,
        residuals=residuals,
        clean_beam=clean_beam,
        peak_history=peak_history,
        n_iter=len(peak_history),
    )


def plot_clean(
    result: CleanResult,
    extent: list[float],
    save: str | Path | None = None,
) -> plt.Figure:
    """Plot CLEAN results: restored image, residuals, and convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Restored image
    vmax = np.percentile(result.restored, 99.5)
    im1 = axes[0].imshow(
        result.restored, origin="lower", extent=extent,
        cmap="afmhot", vmin=0, vmax=vmax,
    )
    axes[0].set_xlabel("Relative RA (μas)")
    axes[0].set_ylabel("Relative Dec (μas)")
    axes[0].set_title("CLEAN Restored Image — M87*")
    axes[0].invert_xaxis()
    fig.colorbar(im1, ax=axes[0], label="Jy/beam", shrink=0.8)

    # Residuals
    rlim = np.percentile(np.abs(result.residuals), 99)
    im2 = axes[1].imshow(
        result.residuals, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-rlim, vmax=rlim,
    )
    axes[1].set_xlabel("Relative RA (μas)")
    axes[1].set_ylabel("Relative Dec (μas)")
    axes[1].set_title("CLEAN Residuals")
    axes[1].invert_xaxis()
    fig.colorbar(im2, ax=axes[1], label="Jy/beam", shrink=0.8)

    # Convergence
    axes[2].semilogy(result.peak_history, color="steelblue", lw=0.8)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Peak Residual")
    axes[2].set_title(f"Convergence ({result.n_iter} iterations)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")

    return fig
