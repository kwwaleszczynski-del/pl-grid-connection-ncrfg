"""Polish seasonal load, solar PV, and onshore wind profiles for MV/110 kV grid connection studies.

  - PV: hourly PVGIS 5.3 output (PVGIS-SARAH3 radiation, ERA5 temperature) at 52.1 N, 19.4 E (central Poland) for
    a fixed, south-facing c-Si array at 40 deg tilt (the PVGIS optimum) with 14 % system losses, 2005-2023
    (seriescalc, queried 2026-10-31, stored in data/). The study year is PV_YEAR, the most recent complete year;
    its DC output is converted to AC per MW of inverter (connection) capacity with an assumed DC/AC ratio of 1.2
    and inverter clipping at 1.0 pu, so pv_cf is per MW of AC capacity.
  - Wind: synthetic, 2,500 full-load hours (capacity factor ~28.5 %), an assumed round value of the order of the
    Polish onshore fleet average; to be verified against ARE/PSE statistics for the year of interest. Hourly wind
    speed is an AR(1) Gaussian process mapped to a seasonal Weibull marginal and passed through a generic
    3-12-25 m/s turbine power curve.
  - Load: synthetic seasonal and daily shape.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


PVGIS_FILE = Path(__file__).parent / "data" / "pvgis_sarah3_52.1N_19.4E_40deg.csv.gz"
PV_YEAR: int = 2023
PV_DC_AC_RATIO: float = 1.2
WIND_FLH: float = 2500.0


@lru_cache(maxsize=1)
def load_pvgis_hourly() -> pd.Series:
    """PVGIS hourly DC output in kW per kWp, 2005-2023, indexed by the UTC start of each hour."""
    df = pd.read_csv(PVGIS_FILE)
    return pd.Series(df["p_w_per_kwp"].to_numpy() / 1000.0, index=pd.DatetimeIndex(pd.to_datetime(df["time_utc"])),
                     name="pv_dc_kw_per_kwp")


def pv_dc_year(pv_year: int = PV_YEAR) -> np.ndarray:
    """8,760 hourly DC values (kW per kWp) of one PVGIS year; 29 February is dropped in leap years."""
    s = load_pvgis_hourly()
    s = s[(s.index.year == pv_year) & ~((s.index.month == 2) & (s.index.day == 29))]
    assert len(s) == 8760
    return s.to_numpy()


def generate_annual_polish_profiles(year: int = 2026, seed: int = 2026, pv_year: int = PV_YEAR) -> pd.DataFrame:
    """Generate an 8,760-hour Polish MV load (synthetic), solar PV (PVGIS) and onshore wind (synthetic) profile.

    Captures the operating regimes that matter for Polish DSO 15 kV and GPZ 110/15 kV studies:
      - Sunny May/June midday minimum load + peak PV export (voltage rise & reverse flow peak)
      - Windy winter/autumn nights with high onshore wind (cable pooling & N-1 backfeed)
      - Winter evening peak load + zero PV (voltage drop & forward transformer loading)

    PV is the PVGIS year ``pv_year`` laid on the calendar of ``year`` hour by hour (both have 8,760 hours).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="1h", tz="UTC")
    n = len(idx)
    ts_pl = idx.tz_convert("Europe/Warsaw")
    doy = np.asarray(idx.day_of_year, dtype=float)
    hour_local = np.asarray(ts_pl.hour, dtype=float)
    dow = np.asarray(ts_pl.day_of_week, dtype=int)

    # The former synthetic PV profile drew its cloudiness from the same generator before the wind; the draws are
    # kept so the seeded wind series is unchanged
    n_days = int(np.ceil(n / 24)) + 1
    rng.beta(2.6, 1.35, size=n_days)
    rng.normal(0.0, 0.05, size=n)
    pv_cf = np.minimum(PV_DC_AC_RATIO * pv_dc_year(pv_year), 1.0)

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
