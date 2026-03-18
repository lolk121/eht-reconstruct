"""
Phase 1: Load and explore EHT visibility data.
 
Usage:
    python explore.py data/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits
 
Downloads the data from:
    git clone https://github.com/eventhorizontelescope/2019-D01-01.git
    cd 2019-D01-01
    tar xzf uvfits/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits.tgz
 
Then copy the .uvfits file into data/
"""
 
import sys
from pathlib import Path
 
from src.parse import load_uvfits, summary
from src.plot import (
    plot_uv_coverage,
    plot_amplitude_vs_uvdist,
    plot_phase_vs_uvdist,
    plot_amplitude_vs_time,
)
 
 
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
 
    print("\nGenerating plots...")
 
    outdir = Path("data")
    outdir.mkdir(exist_ok=True)
 
    plot_uv_coverage(obs, save=outdir / "uv_coverage.png")
    plot_amplitude_vs_uvdist(obs, save=outdir / "amp_vs_uvdist.png")
    plot_phase_vs_uvdist(obs, save=outdir / "phase_vs_uvdist.png")
    plot_amplitude_vs_time(obs, save=outdir / "amp_vs_time.png")
 
    print("\nDone! Check data/ for plots.")
 
 
if __name__ == "__main__":
    main()
 
