"""
Phases 1-4: Load EHT data, explore, dirty image, CLEAN, and MEM reconstruction.

Usage:
    python explore.py data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits
"""

import sys
from pathlib import Path

import numpy as np

from src.parse import load_uvfits, summary
from src.plot import (
    plot_uv_coverage,
    plot_amplitude_vs_uvdist,
    plot_phase_vs_uvdist,
    plot_amplitude_vs_time,
)
from src.dirty import make_dirty_image, plot_dirty
from src.clean import hogbom_clean, plot_clean
from src.mem import mem_reconstruct, plot_mem


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore.py <path-to-uvfits>")
        print()
        print("Example:")
        print("  python explore.py data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits")
        sys.exit(1)

    path = Path(sys.argv[1])
    print(f"Loading {path}...\n")

    obs = load_uvfits(path)
    summary(obs)

    outdir = Path("data")
    outdir.mkdir(exist_ok=True)

    # Phase 1 — diagnostic plots
    print("\n--- Phase 1: Diagnostic plots ---")
    plot_uv_coverage(obs, save=outdir / "uv_coverage.png")
    plot_amplitude_vs_uvdist(obs, save=outdir / "amp_vs_uvdist.png")
    plot_phase_vs_uvdist(obs, save=outdir / "phase_vs_uvdist.png")
    plot_amplitude_vs_time(obs, save=outdir / "amp_vs_time.png")

    # Phase 2 — dirty image
    print("\n--- Phase 2: Dirty image ---")
    dirty, beam, extent = make_dirty_image(obs, npix=256, fov_uas=200.0)
    plot_dirty(dirty, beam, extent, save=outdir / "dirty_image.png")

    print(f"\nDirty image peak: {dirty.max():.4f}")
    print(f"Dirty beam sidelobe level: {np.sort(np.abs(beam.ravel()))[-2]:.3f}")

    # Phase 3 — CLEAN
    print("\n--- Phase 3: CLEAN reconstruction ---")
    result = hogbom_clean(dirty, beam, n_iter=3000, gain=0.05, threshold=0.0001)
    plot_clean(result, extent, save=outdir / "clean_image.png")

    # Phase 4a — Maximum Entropy Method (initialised from CLEAN)
    print("\n--- Phase 4a: MEM reconstruction ---")
    mem_result = mem_reconstruct(obs, npix=128, fov_uas=200.0, n_iter=1000,
                                 init_image=result.restored)
    mem_extent = [-100, 100, -100, 100]
    plot_mem(mem_result, mem_extent, save=outdir / "mem_image.png")

    print("\nDone! Check data/ for all plots.")


if __name__ == "__main__":
    main()
