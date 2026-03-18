"""
UVFITS parser for EHT visibility data.
 
Reads calibrated EHT UVFITS files and extracts:
- Complex visibilities
- (u, v) baseline coordinates in wavelengths
- Per-visibility noise estimates
- Station/antenna metadata
- Timestamps
 
UVFITS format notes:
- Uses FITS "random groups" format (each group = one visibility measurement)
- Group parameters: UU, VV, WW (baseline coords in seconds), DATE, BASELINE
- BASELINE encoding: ant1 * 256 + ant2  (for < 256 antennas)
- u, v in seconds → multiply by frequency to get wavelengths
- Weights → sigma = 1 / sqrt(weight)
 
Usage:
    from src.parse import load_uvfits, summary
    obs = load_uvfits("data/M87_HI_hops_netcal.uvfits")
    summary(obs)
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
 
 
@dataclass
class Observation:
    """Container for one EHT observation."""
 
    u: NDArray[np.float64]           # (N,) baseline u-coords in wavelengths
    v: NDArray[np.float64]           # (N,) baseline v-coords in wavelengths
    vis: NDArray[np.complex128]      # (N,) complex visibilities
    sigma: NDArray[np.float64]       # (N,) thermal noise (1-sigma)
    time_mjd: NDArray[np.float64]    # (N,) timestamps (MJD)
    ant1: NDArray[np.int32]          # (N,) first antenna index
    ant2: NDArray[np.int32]          # (N,) second antenna index
    stations: list[str]              # station names from antenna table
    freq_hz: float                   # observing frequency in Hz
    source: str                      # source name (e.g. "M87")
 
    @property
    def n_vis(self) -> int:
        return len(self.vis)
 
    @property
    def n_stations(self) -> int:
        return len(self.stations)
 
    @property
    def baselines(self) -> list[tuple[str, str]]:
        """Unique baseline pairs as (station, station) tuples."""
        pairs = set()
        for a1, a2 in zip(self.ant1, self.ant2):
            pairs.add((self.stations[a1], self.stations[a2]))
        return sorted(pairs)
 
    @property
    def u_glambda(self) -> NDArray[np.float64]:
        """u-coordinates in giga-wavelengths (for plotting)."""
        return self.u / 1e9
 
    @property
    def v_glambda(self) -> NDArray[np.float64]:
        """v-coordinates in giga-wavelengths (for plotting)."""
        return self.v / 1e9
 
 
def load_uvfits(path: str | Path) -> Observation:
    """
    Load an EHT UVFITS file into an Observation.
 
    Parameters
    path : str or Path
        Path to a .uvfits file.
 
    Returns
    -------
    Observation
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UVFITS file not found: {path}")
 
    with fits.open(path) as hdul:
        # Primary HDU contains the visibility data in random-groups format
        primary = hdul[0]
        header = primary.header
        data = primary.data
 
        # Frequency is encoded in the header via CRVAL4 (or similar axis)
        # The exact keyword depends on the UVFITS variant
        freq_hz = _extract_frequency(header)
 
        # UU and VV are baseline coordinates in seconds
        # Multiply by freq to get wavelengths
        uu_sec = data.par("UU")
        vv_sec = data.par("VV")
        u = uu_sec * freq_hz
        v = vv_sec * freq_hz
 
        # DATE Modified Julian Date
        # Some files split across DATE and DATE2
        time_mjd = data.par("DATE")
        try:
            time_mjd = time_mjd + data.par("DATE2")
        except KeyError:
            pass
 
        # BASELINE encoded as ant1 * 256 + ant2
        baseline_codes = data.par("BASELINE").astype(np.int32)
        ant1 = (baseline_codes // 256) - 1  # 0-indexed
        ant2 = (baseline_codes % 256) - 1
 
        # Data array shape: (n_vis, 1, 1, n_freq, n_stokes, 3)
        #last dim: [real, imag, weight]
        # Squeeze out degenerate axes first
        vis_data = data.data.squeeze()
 
        # Handle different possible shapes after squeeze
        # We want to get to shape (n_vis, 3) or average over freq/stokes
        if vis_data.ndim == 2 and vis_data.shape[-1] == 3:
            # Already (n_vis, 3) — single freq, single stokes
            real = vis_data[:, 0]
            imag = vis_data[:, 1]
            weights = vis_data[:, 2]
        elif vis_data.ndim == 3 and vis_data.shape[-1] == 3:
            # (n_vis, n_stokes, 3) — average over stokes
            real = np.mean(vis_data[:, :, 0], axis=1)
            imag = np.mean(vis_data[:, :, 1], axis=1)
            weights = np.mean(vis_data[:, :, 2], axis=1)
        elif vis_data.ndim == 4 and vis_data.shape[-1] == 3:
            # (n_vis, n_freq, n_stokes, 3) — average over freq and stokes
            real = np.mean(vis_data[:, :, :, 0], axis=(1, 2))
            imag = np.mean(vis_data[:, :, :, 1], axis=(1, 2))
            weights = np.mean(vis_data[:, :, :, 2], axis=(1, 2))
        else:
            raise ValueError(
                f"Unexpected visibility data shape after squeeze: {vis_data.shape}. "
                f"Expected last dimension to be 3 (real, imag, weight)."
            )
 
        vis = real + 1j * imag
 
        # Weights -> sigma: sigma = 1 / sqrt(weight) for weight > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(weights > 0, 1.0 / np.sqrt(weights), np.inf)
 
        stations = _extract_stations(hdul)
 
        source = header.get("OBJECT", "UNKNOWN")
 
    # Remove entries with zero or negative weights, or auto-correlations
    valid = (weights > 0) & (ant1 != ant2)
    u = u[valid]
    v = v[valid]
    vis = vis[valid]
    sigma = sigma[valid]
    time_mjd = time_mjd[valid]
    ant1 = ant1[valid]
    ant2 = ant2[valid]
 
    return Observation(
        u=u.astype(np.float64),
        v=v.astype(np.float64),
        vis=vis.astype(np.complex128),
        sigma=sigma.astype(np.float64),
        time_mjd=time_mjd.astype(np.float64),
        ant1=ant1.astype(np.int32),
        ant2=ant2.astype(np.int32),
        stations=stations,
        freq_hz=float(freq_hz),
        source=source,
    )
 
 
def _extract_frequency(header: fits.Header) -> float:
    """
    Pull the reference frequency from the UVFITS header.
 
    The frequency axis is typically CTYPE4='FREQ' with CRVAL4 = freq in Hz.
    But the axis number varies across files, so we search for it.
    """
    for i in range(1, header.get("NAXIS", 7) + 1):
        ctype = header.get(f"CTYPE{i}", "")
        if "FREQ" in ctype.upper():
            return float(header[f"CRVAL{i}"])
 
    if "RESTFREQ" in header:
        return float(header["RESTFREQ"])
    if "FREQ" in header:
        return float(header["FREQ"])
 
    raise ValueError("Could not find frequency in UVFITS header")
 
 
def _extract_stations(hdul: fits.HDUList) -> list[str]:
    """
    Extract station names from the AIPS AN (antenna) table.
 
    Falls back to generic numbered names if no antenna table found.
    """
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.name.upper() == "AIPS AN":
            names = hdu.data["ANNAME"]
            # Clean whitespace
            return [name.strip() for name in names]
 
    primary = hdul[0]
    baselines = primary.data.par("BASELINE").astype(np.int32)
    max_ant = max(baselines // 256)
    return [f"ANT{i}" for i in range(max_ant + 1)]
 
 
def summary(obs: Observation) -> None:
    """Print a human-readable summary of the observation."""
    print(f"Source:       {obs.source}")
    print(f"Frequency:    {obs.freq_hz / 1e9:.1f} GHz")
    print(f"Stations:     {obs.n_stations} — {', '.join(obs.stations)}")
    print(f"Visibilities: {obs.n_vis:,}")
    print(f"Baselines:    {len(obs.baselines)}")
    print(f"Time range:   MJD {obs.time_mjd.min():.4f} — {obs.time_mjd.max():.4f}")
    print(f"uv range:     {np.hypot(obs.u, obs.v).min()/1e9:.2f} — {np.hypot(obs.u, obs.v).max()/1e9:.2f} Gλ")
    print(f"Median SNR:   {np.median(np.abs(obs.vis) / obs.sigma):.1f}")
