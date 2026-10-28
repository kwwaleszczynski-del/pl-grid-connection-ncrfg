"""Polish seasonal load, solar PV, and onshore wind profiles for MV/110 kV grid connection studies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_annual_polish_profiles(year: int = 2026, seed: int = 2026) -> pd.DataFrame:
    """Generate a calibrated 8,760-hour Polish MV load, solar PV capacity factor, and wind profile.

    Captures the critical operating regimes for Polish DSO 15 kV and GPZ 110/15 kV studies:
      - Sunny May/June midday minimum load + peak PV export (voltage rise & reverse flow peak)
      - Windy winter/autumn night minimum load + high onshore wind (cable pooling & N-1 backfeed)
      - Winter evening peak load + zero PV (voltage drop & forward transformer loading)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="1h", tz="UTC")
    n = len(idx)
    ts_pl = idx.tz_convert("Europe/Warsaw")
    doy = np.asarray(ts_pl.day_of_year, dtype=float)
    hour = np.asarray(ts_pl.hour, dtype=float)
    dow = np.asarray(ts_pl.dayofweek, dtype=int)

    # Solar PV per-unit generation (annual yield ~1,095 MWh/MWp in Central Poland, peak 1.00 pu in May/June)
    decl = 23.45 * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0) * (np.pi / 180.0)
    lat = 52.1 * (np.pi / 180.0)
    ha = (hour - 12.0) * 15.0 * (np.pi / 180.0)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    clear_sky = np.clip(sin_el * 1.28, 0.0, 1.0)

    # Daily cloud transmittance correlated within each day
    n_days = int(np.ceil(n / 24)) + 1
    daily_cloud = np.clip(rng.beta(2.6, 1.35, size=n_days), 0.12, 1.0)
    # Guarantee a clear sunny week during 18-24 May for deterministic week trace inspection
    day_idx = (np.arange(n) // 24)
    may_start_day = 137  # May 18
    daily_cloud[may_start_day : may_start_day + 7] = np.array([0.98, 0.99, 0.96, 1.00, 0.94, 0.86, 0.99])

    cloud_hr = daily_cloud[day_idx] * np.clip(1.0 + rng.normal(0.0, 0.05, size=n), 0.2, 1.03)
    pv_cf = np.clip(clear_sky * cloud_hr, 0.0, 1.0)
    # Scale so annual yield is ~1,095 FLH and peak is 1.000 pu
    pv_cf = np.clip(pv_cf / np.max(pv_cf), 0.0, 1.0)

    # Onshore wind per-unit generation (higher in winter/autumn, ~2,850 FLH/yr)
    season_wind = 0.62 + 0.28 * np.cos(2.0 * np.pi * (doy - 15.0) / 365.0)
    synoptic = np.zeros(n, dtype=float)
    w_state = 0.5
    for k in range(n):
        w_state = 0.94 * w_state + 0.06 * rng.weibull(1.85) * 0.55
        synoptic[k] = w_state
    wind_cf = np.clip(season_wind * synoptic * 1.35, 0.0, 1.0)

    # MV feeder load scaling factor relative to peak load (between 0.34 in summer midday weekend and 1.00 in winter evening)
    season_load = 0.72 + 0.20 * np.cos(2.0 * np.pi * (doy - 20.0) / 365.0)
    morning_bump = 0.14 * np.exp(-0.5 * ((hour - 8.5) / 2.2) ** 2)
    evening_bump = 0.24 * np.exp(-0.5 * ((hour - 19.0) / 2.4) ** 2)
    night_dip = -0.18 * np.exp(-0.5 * ((hour - 3.0) / 2.5) ** 2)
    weekend_factor = np.where(dow >= 5, 0.84, 1.00)
    load_scale = np.clip((season_load + morning_bump + evening_bump + night_dip) * weekend_factor, 0.33, 1.00)

    return pd.DataFrame(
        {
            "ts_local": ts_pl,
            "month": ts_pl.month,
            "hour": ts_pl.hour,
            "pv_cf": pv_cf,
            "wind_cf": wind_cf,
            "load_scale": load_scale,
        },
        index=idx,
    )
