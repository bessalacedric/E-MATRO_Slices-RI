"""
Data Loader
===========
Loads real 5G BS measurement CSV files (G-NetTrack format) and converts
them into a (T × N_rus) traffic matrix used by OCloudEnv.

Expected CSV columns:
    Timestamp   — datetime string
    CellID      — cell identifier
    UL_bitrate  — uplink bitrate in kbps
    (optional: RSRP, CQI, RSRQ, DL_bitrate, SNR)

Usage:
    from utils.data_loader import load_traffic
    traffic = load_traffic("data/bs_traces", num_rus=6)
    # traffic.shape = (T, 6)  values in Mbps
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def load_traffic(
    data_dir:      str,
    num_rus:       int,
    resample:      str   = "1min",
    peak_mbps:     float = 100.0,
    normalize:     bool  = False,
) -> Tuple[Optional[np.ndarray], list]:
    """
    Load all CSV files in data_dir, aggregate UL_bitrate per cell per minute.

    Parameters
    ----------
    data_dir   : path to folder containing one CSV per BS
    num_rus    : number of RUs to use (takes first N files found)
    resample   : pandas resample frequency string
    peak_mbps  : used for normalization when normalize=True
    normalize  : if True, divide by peak_mbps so values are in [0,1]

    Returns
    -------
    traffic    : np.ndarray shape (T, num_rus) in Mbps (or [0,1] if normalized)
                 None if no files found
    cell_ids   : list of cell IDs in column order
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas not installed — run: pip install pandas")
        return None, []

    files = sorted(Path(data_dir).glob("*.csv")) if os.path.isdir(data_dir) else []
    if not files:
        log.warning(f"No CSV files found in {data_dir}")
        return None, []

    selected = files[:num_rus]
    log.info(f"Loading {len(selected)} BS files from {data_dir}")

    series = {}
    for f in selected:
        try:
            df = pd.read_csv(f)

            # ── Timestamp parsing: try common formats before falling back ──
            ts_raw = df["Timestamp"]
            parsed = None
            for fmt in ("%Y.%m.%d %H.%M.%S",   # 2019.12.14 10.16.30
                        "%Y-%m-%d %H:%M:%S",     # 2019-12-14 10:16:30
                        "%Y/%m/%d %H:%M:%S",     # 2019/12/14 10:16:30
                        "%Y.%m.%d_%H.%M.%S",     # 2019.12.14_10.16.30
                        "%d/%m/%Y %H:%M:%S",     # 14/12/2019 10:16:30
                        "%d/%m/%Y %H:%M:%S",     # 14/12/2019 10:16:30
                        None):                   # last resort: let pandas infer
                try:
                    if fmt is None:
                        parsed = pd.to_datetime(ts_raw, errors="coerce")
                    else:
                        parsed = pd.to_datetime(ts_raw, format=fmt, errors="coerce")
                    valid = parsed.notna().sum()
                    if valid > len(ts_raw) * 0.5:  # >50% parsed → accept
                        log.info(f"  {f.name}: format='{fmt}', "
                                 f"{valid}/{len(ts_raw)} timestamps parsed")
                        break
                except Exception:
                    continue

            if parsed is None or parsed.notna().sum() == 0:
                log.warning(f"  Could not parse timestamps in {f.name} — skipping")
                continue

            df["Timestamp"] = parsed
            df = df.dropna(subset=["Timestamp", "UL_bitrate"])
            df = df[df["UL_bitrate"] >= 0]

            # Guard: skip if dataframe is empty after filtering
            if df.empty:
                log.warning(f"  {f.name}: no valid rows after filtering — skipping")
                continue

            cell_id = str(df["CellID"].iloc[0]) if "CellID" in df.columns \
                      else f.stem

            # Set DatetimeIndex then resample to regular 1-min grid
            agg = (
                df.set_index("Timestamp")["UL_bitrate"]
                  .resample(resample)
                  .sum()      # total UL load across all UEs in this cell
                  .ffill()
                  .bfill()
            )
            series[cell_id] = agg / 1e3    # kbps → Mbps
            log.info(f"  {f.name}: {len(agg)} steps, "
                     f"peak={agg.max():.1f} Mbps, mean={agg.mean():.1f} Mbps")

        except Exception as e:
            log.warning(f"  Could not load {f.name}: {e}")

    if not series:
        return None, []

    # Align all series on a common time index
    traffic = (
        pd.DataFrame(series)
          .sort_index()
          .interpolate(method="time")
          .fillna(0.0)
    )

    # Pad to num_rus columns if fewer files were loaded
    while traffic.shape[1] < num_rus:
        col = f"pad_{traffic.shape[1]}"
        traffic[col] = traffic.iloc[:, 0].values
        log.info(f"  Padding with copy of column 0 as '{col}'")

    traffic = traffic.iloc[:, :num_rus]
    cell_ids = list(traffic.columns)

    arr = traffic.values.astype(np.float32)
    if normalize:
        arr = np.clip(arr / peak_mbps, 0.0, 1.0)

    # Tile to reach at least 1440 steps (1 full day equivalent)
    min_steps = 1440
    if len(arr) < min_steps:
        repeats = int(np.ceil(min_steps / len(arr)))
        arr = np.tile(arr, (repeats, 1))[:min_steps]
        log.info(f"Traffic tiled x{repeats} -> {len(arr)} steps")

    log.info(f"Traffic matrix: shape={arr.shape}, "
             f"max={arr.max():.1f}, mean={arr.mean():.1f} "
             f"{'(normalised)' if normalize else 'Mbps'}")

    return arr, cell_ids


def make_synthetic_traffic(
    T:      int   = 1440,
    N:      int   = 6,
    seed:   int   = 42,
) -> np.ndarray:
    """
    Diurnal + AR(1) synthetic traffic (Mbps).
    Two daily peaks: morning (~09:00) and evening (~20:00).
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(T, dtype=np.float32)
    out = np.zeros((T, N), dtype=np.float32)

    for n in range(N):
        amplitude = rng.uniform(0.7, 1.3)
        diurnal   = amplitude * (
            5.0
            + 40.0 * np.exp(-((t -  540) ** 2) / (2 * 120.0**2))
            + 60.0 * np.exp(-((t - 1200) ** 2) / (2 * 150.0**2))
        )
        noise = np.zeros(T, dtype=np.float32)
        for i in range(1, T):
            noise[i] = 0.85 * noise[i-1] + float(rng.normal(0, 3.0))
        out[:, n] = np.clip(diurnal + noise, 0.1, 100.0)

    return out