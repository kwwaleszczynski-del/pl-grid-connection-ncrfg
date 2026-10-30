"""Synthetic Polish seasonal load, solar PV, and onshore wind profiles for MV/110 kV grid connection studies.

The profiles are synthetic (no measured data) and calibrated to two annual totals:
  - PV: 1,086 kWh/kWp, the PVGIS-SARAH3 yield at 52.1 N, 19.4 E (central Poland) for a fixed, south-facing
    c-Si array at the PVGIS optimal tilt (40 deg) with 14 % system losses (PVGIS PVcalc, queried 2026-10-30).
    The DC profile is converted to AC per MW of inverter (connection) capacity with an assumed DC/AC ratio of
    1.2 and inverter clipping at 1.0 pu, so pv_cf is per MW of AC capacity.
  - Wind: 2,500 full-load hours (capacity factor ~28.5 %), an assumed round value of the order of the Polish
    onshore fleet average; to be verified against ARE/PSE statistics for the year of interest. Hourly wind
    speed is an AR(1) Gaussian process mapped to a seasonal Weibull marginal and passed through a generic
    3-12-25 m/s turbine power curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


PV_YIELD_KWH_PER_KWP: float = 1086.0
PV_DC_AC_RATIO: float = 1.2
WIND_FLH: float = 2500.0
SITE_LAT_DEG: float = 52.1
SITE_LON_DEG: float = 19.4


def _scale_to_annual_flh(shape: np.ndarray, target_flh: float) -> np.ndarray:
    """Return clip(k * shape, 0, 1) with k found by bisection so that the annual sum equals target_flh."""
    lo, hi = 0.0, 1.0
    while np.sum(np.clip(hi * shape, 0.0, 1.0)) < target_flh:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sum(np.clip(mid * shape, 0.0, 1.0)) < target_flh:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * shape, 0.0, 1.0)


def generate_annual_polish_profiles(year: int = 2026, seed: int = 2026) -> pd.DataFrame:
    """Generate an 8,760-hour synthetic Polish MV load, solar PV and onshore wind profile.

    Captures the operating regimes that matter for Polish DSO 15 kV and GPZ 110/15 kV studies:
      - Sunny May/June midday minimum load + peak PV export (voltage rise & reverse flow peak)
      - Windy winter/autumn nights with high onshore wind (cable pooling & N-1 backfeed)
      - Winter evening peak load + zero PV (voltage drop & forward transformer loading)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="1h", tz="UTC")
    n = len(idx)
    ts_pl = idx.tz_convert("Europe/Warsaw")
    doy = np.asarray(idx.day_of_year, dtype=float)
    hour_local = np.asarray(ts_pl.hour, dtype=float)
    dow = np.asarray(ts_pl.day_of_week, dtype=int)

    # Solar elevation from local mean solar time (UTC + longitude / 15 h; equation of time neglected),
    # evaluated at the middle of each hourly interval
    solar_time_h = np.asarray(idx.hour, dtype=float) + 0.5 + SITE_LON_DEG / 15.0
    decl = 23.45 * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0) * (np.pi / 180.0)
    lat = SITE_LAT_DEG * (np.pi / 180.0)
    ha = (solar_time_h - 12.0) * 15.0 * (np.pi / 180.0)
    sin_el = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha), 0.0, None)

    # Daily cloud transmittance correlated within each day
    n_days = int(np.ceil(n / 24)) + 1
    daily_cloud = np.clip(rng.beta(2.6, 1.35, size=n_days), 0.12, 1.0)
    # Impose a clear week on 18-24 May (illustrative sunny-week trace); it is part of the calibrated year
    day_idx = (np.arange(n) // 24)
    may_start_day = 137  # May 18
    daily_cloud[may_start_day : may_start_day + 7] = np.array([0.98, 0.99, 0.96, 1.00, 0.94, 0.86, 0.99])

    cloud_hr = daily_cloud[day_idx] * np.clip(1.0 + rng.normal(0.0, 0.05, size=n), 0.2, 1.03)
    pv_dc_per_kwp = _scale_to_annual_flh(sin_el * cloud_hr, PV_YIELD_KWH_PER_KWP)
    pv_cf = np.minimum(PV_DC_AC_RATIO * pv_dc_per_kwp, 1.0)

    # Onshore wind: AR(1) standard-normal synoptic state -> Weibull(k = 2) hub-height speed with a seasonal
    # scale (winter maximum) -> generic power curve (cut-in 3, rated 12, cut-out 25 m/s); the Weibull scale
    # is set by bisection so the annual yield equals WIND_FLH
    z = np.zeros(n, dtype=float)
    z_state = 0.0
    eps = rng.normal(0.0, 1.0, size=n)
    for k in range(n):
        z_state = 0.97 * z_state + np.sqrt(1.0 - 0.97 ** 2) * eps[k]
        z[k] = z_state
    season_wind = 1.0 + 0.18 * np.cos(2.0 * np.pi * (doy - 15.0) / 365.0)
    weibull_unit = (-np.log(1.0 - np.clip(ndtr(z), 1e-12, 1.0 - 1e-12))) ** (1.0 / 2.0)

    def _wind_power(scale_m_s: float) -> np.ndarray:
        v = scale_m_s * season_wind * weibull_unit
        p = np.clip((v ** 3 - 3.0 ** 3) / (12.0 ** 3 - 3.0 ** 3), 0.0, 1.0)
        return np.where(v >= 25.0, 0.0, p)

    lo, hi = 1.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sum(_wind_power(mid)) < WIND_FLH else (lo, mid)
    wind_cf = _wind_power(hi)

    # MV feeder load scaling factor relative to peak load (0.33 on summer weekend nights .. 1.00 on winter evenings)
    season_load = 0.72 + 0.20 * np.cos(2.0 * np.pi * (doy - 20.0) / 365.0)
    morning_bump = 0.14 * np.exp(-0.5 * ((hour_local - 8.5) / 2.2) ** 2)
    evening_bump = 0.24 * np.exp(-0.5 * ((hour_local - 19.0) / 2.4) ** 2)
    night_dip = -0.18 * np.exp(-0.5 * ((hour_local - 3.0) / 2.5) ** 2)
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
