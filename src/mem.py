"""
Phase 4a: Maximum Entropy Method (MEM) image reconstruction.

Uses multiplicative gradient updates with CLEAN initialisation.

Usage:
    from src.parse import load_uvfits
    from src.mem import mem_reconstruct, plot_mem

    obs = load_uvfits("data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits")
    result = mem_reconstruct(obs, npix=128, fov_uas=200.0)
    plot_mem(result, [-100, 100, -100, 100], save="data/mem_image.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.parse import Observation


UAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1e6)


@dataclass
class MEMResult:
    """Output of MEM reconstruction."""
    image: NDArray[np.float64]
    chi2_history: list[float]
    entropy_history: list[float]
    n_iter: int


class GriddedOperator:
    """
    Forward/adjoint operators via gridded FFT.
    Adjoint of FFT = N^2 * IFFT.
    """

    def __init__(self, obs: Observation, npix: int, fov_uas: float):
        self.npix = npix
        fov_rad = fov_uas * UAS_TO_RAD
        pixel_rad = fov_rad / npix
        du = 1.0 / (npix * pixel_rad)

        u_idx = np.round(obs.u / du).astype(np.int64) + npix // 2
        v_idx = np.round(obs.v / du).astype(np.int64) + npix // 2

        valid = (u_idx >= 0) & (u_idx < npix) & (v_idx >= 0) & (v_idx < npix)
        self.u_idx = u_idx[valid]
        self.v_idx = v_idx[valid]
        self.vis_obs = obs.vis[valid]
        self.sigma = obs.sigma[valid]
        self.n_vis = int(valid.sum())
        self.inv_sigma2 = 1.0 / self.sigma**2

        print(f"GriddedOperator: {self.n_vis}/{len(obs.vis)} visibilities on grid")

    def forward(self, image: NDArray[np.float64]) -> NDArray[np.complex128]:
        ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        return ft[self.v_idx, self.u_idx]

    def adjoint(self, vis: NDArray[np.complex128]) -> NDArray[np.float64]:
        grid = np.zeros((self.npix, self.npix), dtype=np.complex128)
        np.add.at(grid, (self.v_idx, self.u_idx), vis)
        return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid)))) * self.npix**2

    def chi_squared(self, image: NDArray[np.float64]) -> float:
        residuals = self.vis_obs - self.forward(image)
        return float(np.sum(np.abs(residuals)**2 * self.inv_sigma2) / self.n_vis)

    def chi2_gradient(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        residuals = (self.forward(image) - self.vis_obs) * self.inv_sigma2
        return (2.0 / self.n_vis) * self.adjoint(residuals)


def _entropy(image, prior):
    valid = image > 0
    return -float(np.sum(image[valid] * np.log(image[valid] / prior[valid])))


def _entropy_gradient(image, prior):
    grad = np.zeros_like(image)
    valid = image > 0
    grad[valid] = -np.log(image[valid] / prior[valid]) - 1.0
    return grad


def mem_reconstruct(
    obs: Observation,
    npix: int = 128,
    fov_uas: float = 200.0,
    n_iter: int = 1000,
    target_chi2: float = 1.5,
    init_image: NDArray[np.float64] | None = None,
) -> MEMResult:
    """
    MEM reconstruction with multiplicative gradient updates.
    """
    op = GriddedOperator(obs, npix, fov_uas)

    uvdist = np.hypot(obs.u, obs.v)
    short_bl = uvdist < np.percentile(uvdist, 10)
    total_flux = float(np.median(np.abs(obs.vis[short_bl]))) if short_bl.sum() > 0 else 0.5

    prior_floor = total_flux / npix**2 * 0.01

    if init_image is not None:
        if init_image.shape[0] != npix:
            from scipy.ndimage import zoom
            image = zoom(init_image, npix / init_image.shape[0], order=1)
        else:
            image = init_image.copy()
        image = np.maximum(image, 0)
        image = image / image.sum() * total_flux
        image = np.maximum(image, prior_floor)
        print(f"  Initialised from provided image, rescaled to {total_flux:.4f} Jy")
    else:
        yy, xx = np.mgrid[:npix, :npix]
        cy, cx = npix // 2, npix // 2
        image = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * (npix / 8)**2))
        image = image / image.sum() * total_flux
        image = np.maximum(image, prior_floor)

    prior = np.full((npix, npix), total_flux / npix**2)

    chi2_history = []
    entropy_history = []

    chi2_init = op.chi_squared(image)
    model_init = op.forward(image)
    print(f"MEM: npix={npix}, max {n_iter} iters, target chi2={target_chi2}")
    print(f"  Total flux = {total_flux:.4f} Jy")
    print(f"  |V_obs| median = {np.median(np.abs(op.vis_obs)):.6f}")
    print(f"  |V_model| median = {np.median(np.abs(model_init)):.6f}")
    print(f"  Initial chi2 = {chi2_init:.2f}")

    lam = max(0.1, chi2_init / 100.0)
    best_image = image.copy()
    best_chi2 = chi2_init

    for i in range(n_iter):
        grad_chi2 = op.chi2_gradient(image)
        grad_entropy = _entropy_gradient(image, prior)

        grad = grad_entropy - lam * grad_chi2

        step = image * grad
        step_max = np.max(np.abs(step))
        if step_max > 0:
            scale = min(1.0, 0.5 * image.max() / step_max)
            image = image + scale * step

        image = np.maximum(image, prior_floor)
        image *= total_flux / image.sum()

        chi2 = op.chi_squared(image)
        ent = _entropy(image, prior)
        chi2_history.append(chi2)
        entropy_history.append(ent)

        if abs(chi2 - target_chi2) < abs(best_chi2 - target_chi2):
            best_chi2 = chi2
            best_image = image.copy()

        chi2_ratio = chi2 / target_chi2
        if chi2_ratio > 10:
            lam *= 2.0
        elif chi2_ratio > 2:
            lam *= 1.2
        elif chi2_ratio < 0.5:
            lam *= 0.8
        lam = min(lam, 1e12)

        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}: chi2 = {chi2:.2f}, S = {ent:.4f}, lam = {lam:.4e}")

        if 0.8 < chi2_ratio < 1.5 and i > 100:
            recent = chi2_history[-20:]
            if len(recent) == 20 and max(recent) / min(recent) < 1.1:
                print(f"  Converged at iteration {i + 1}: chi2 = {chi2:.2f}")
                break

    print(f"  Best chi2 = {best_chi2:.2f}")

    return MEMResult(
        image=best_image,
        chi2_history=chi2_history,
        entropy_history=entropy_history,
        n_iter=len(chi2_history),
    )


def plot_mem(
    result: MEMResult,
    extent: list[float],
    save: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = np.percentile(result.image, 99.5)
    im = axes[0].imshow(
        result.image, origin="lower", extent=extent,
        cmap="afmhot", vmin=0, vmax=vmax,
    )
    axes[0].set_xlabel("Relative RA (μas)")
    axes[0].set_ylabel("Relative Dec (μas)")
    axes[0].set_title("MEM Reconstruction — M87*")
    axes[0].invert_xaxis()
    fig.colorbar(im, ax=axes[0], label="Jy/pixel", shrink=0.8)

    axes[1].semilogy(result.chi2_history, color="steelblue", lw=1.2)
    axes[1].axhline(1.0, color="red", ls="--", lw=0.8, label="Target χ²=1")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Reduced χ²")
    axes[1].set_title("Data Fidelity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.entropy_history, color="coral", lw=1.2)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Entropy S")
    axes[2].set_title("Entropy")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")

    return fig
