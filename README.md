# eht-reconstruct

Independent image reconstruction pipeline for Event Horizon Telescope (EHT) interferometric data. Reconstructs black hole images from raw VLBI visibility measurements. Built from scratch without existing radio astronomy imaging libraries (hopefully).

---

## Project Plan

### What I'm Building

The EHT captured the first images of black holes (M87\* and Sgr A\*) by combining radio signals from telescopes worldwide. The raw data isn't an image, it's sparse, noisy Fourier domain measurements called *visibilities*. Turning those into an image is an ill posed inverse problem. I'm attempting to build my own pipeline to solve it.

**Pipeline:** UVFITS file -> parse visibilities -> grid onto Fourier plane -> reconstruct image

### Milestones

**Phase 1 - Data ingestion & exploration***
- [done] Parse EHT UVFITS files using astropy (extract visibilities, uv-coordinates, noise estimates)
- [done] Plot uv-coverage to understand which Fourier components are sampled
- [done] Sanity check: visibility amplitudes and phases look reasonable

**Phase 2 - Dirty image baseline**
- [done] Implement inverse NUFFT (irregularly sampled Fourier data -> image) using finufft
- [done] Generate dirty image (naive inverse FFT, heavily artefacted but proves the pipeline works)
- [done] Compute and visualise the dirty beam (point spread function)

**Phase 3 - CLEAN reconstruction**
- [done] Implement Hogbom CLEAN 
- [done] Fit clean beam (Gaussian to PSF main lobe)
- [done] Generate restored image, first real reconstruction of M87\*
- [done] Convergence diagnostics (residual vs iteration)

**Phase 4 - Advanced methods*** 
- [ ] Maximum Entropy Method (smoothest image consistent with data)
- [ ] Compressed sensing via FISTA (L1-regularised sparse recovery)
- [ ] Algorithm comparison with metrics (chi-squared, cross-correlation)

**Phase 5 - Stretch goals**
- [ ] Sgr A\* reconstruction (time variable source much much harder)
- [ ] GPU-accelerated gridding (CuPy or JAX)
- [ ] Self-calibration loop (phase gain corrections using the reconstruction itself)

### Key Technical Challenges

1. **The math** - Fourier transforms on irregular grids, deconvolution, regularisation theory. Graduate level signal processing.
2. **The data format** - UVFITS is an astronomy specific format from the 1980s. Poorly documented, assumes VLBI domain knowledge.
3. **Tuning** - Getting CLEAN/MEM to produce something that looks like M87\* (not noise) requires careful parameter tuning.

### Data Sources

- **M87* 2017 (calibrated visibilities):** https://github.com/eventhorizontelescope/2019-D01-01
- **M87* 2018 (calibrated visibilities):** https://github.com/eventhorizontelescope/2024-D01-01
- **EHT imaging pipelines (reference):** https://github.com/eventhorizontelescope/2019-D01-02

Download UVFITS files and place in `data/`.

### Learning Resources (in order)

1. [EHT Paper IV (2019) — Imaging the M87\* Black Hole](https://doi.org/10.3847/2041-8213/ab0e85) - what methods the EHT team used
2. [Daniel Palumbo's EHT Imaging Tutorial (YouTube)](https://www.youtube.com/watch?v=1vJGMV2MXKM) - accessible walkthrough
4. [Högbom (1974)](https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H) - original CLEAN paper
5. [Cornwell & Evans (1985)](https://doi.org/10.1051/0004-6361:19850109) - MEM for radio imaging
6. [Akiyama et al. (2017)](https://doi.org/10.3847/1538-4357/aa6305) - compressed sensing for VLBI

### Tech Stack

- **Python 3.11+**
- **astropy** - UVFITS parsing
- **finufft** - non-uniform FFT
- **numpy / scipy** - core numerics
- **matplotlib** - visualisation

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/eht-reconstruct.git
cd eht-reconstruct
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT
