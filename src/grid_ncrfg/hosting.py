"""Hosting capacity search, export-cap energy trade-off, co-located BESS dispatch, and GPZ N-1 time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .network import NetworkModel, build_polish_gpz_network
from .ncrfg import (
    CONTROL_REGIMES,
    compute_cos_phi_p_q_mvar,
    make_qu_law,
    qu_q_limit_mvar,
    solve_feeder_with_ncrfg,
)
from .powerflow import PowerFlowResult, solve_ac_powerflow
from .profiles import PV_DC_AC_RATIO, generate_annual_polish_profiles, load_pvgis_hourly, pv_dc_year


V_MAX_PU: float = 1.050
GPZ_EXISTING_GEN_MW: float = 8.0   # assumed PV already connected at the GPZ 15 kV busbar (scenario input)


def find_feeder_hosting_capacity_mw(
    net: NetworkModel,
    feeder_code: str,
    regime_code: str,
    v_max_limit_pu: float = V_MAX_PU,
    loading_limit_pct: float = 100.0,
    p_max_search_mw: float = 40.0,
) -> dict[str, float | str]:
    """Find maximum connectable capacity P_inst [MW] on a feeder by binary search over exact AC power flow."""
    fspec = net.feeders[feeder_code]
    bus_ids = list(fspec.bus_ids)
    br_ids = list(fspec.branch_ids)

    lo, hi = 0.0, p_max_search_mw
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        res = solve_feeder_with_ncrfg(net, feeder_code, mid, regime_code)
        vmax = float(np.max(res.v_mag_pu[bus_ids]))
        imax = float(np.max(res.branch_loading_pct[br_ids]))
        if vmax <= v_max_limit_pu and imax <= loading_limit_pct:
            lo = mid
        else:
            hi = mid

    res_opt = solve_feeder_with_ncrfg(net, feeder_code, lo, regime_code)
    res_hi = solve_feeder_with_ncrfg(net, feeder_code, hi, regime_code)
    vmax_opt = float(np.max(res_opt.v_mag_pu[bus_ids]))
    imax_opt = float(np.max(res_opt.branch_loading_pct[br_ids]))
    vmax_hi = float(np.max(res_hi.v_mag_pu[bus_ids]))
    binding = "PCC voltage (+5.0 %)" if vmax_hi > v_max_limit_pu else "Thermal ampacity (I_dd)"
    q_plant = float(np.sum(res_opt.q_inj_mvar[bus_ids])) + sum(net.buses[b].q_load_min_mvar for b in bus_ids)
    p_loss_kw = float(np.sum(res_opt.branch_p_loss_mw[br_ids]) * 1000.0)

    return {
        "feeder": feeder_code,
        "regime": regime_code,
        "hosting_capacity_mw": float(np.floor(lo * 100.0) / 100.0),  # rounded down: stays feasible
        "v_pcc_pu": round(vmax_opt, 4),
        "max_branch_loading_pct": round(imax_opt, 1),
        "q_net_mvar": round(q_plant, 2),
        "feeder_loss_kw": round(p_loss_kw, 1),
        "binding_constraint": binding,
    }


def solve_gpz_busbar_injection(
    net: NetworkModel,
    p_new_mw: float,
    regime_code: str = "uncontrolled",
    p_existing_mw: float = GPZ_EXISTING_GEN_MW,
    load_scale: float | None = None,
) -> PowerFlowResult:
    """AC power flow with new generation connected at the GPZ 15 kV busbar (bus 1) by a dedicated bay.

    Existing generation p_existing_mw also sits on the busbar at cos phi = 1. Loads are the minimum daytime
    loads, or the peak loads times load_scale when given.
    """
    reg = CONTROL_REGIMES[regime_code]
    n = len(net.buses)
    if load_scale is None:
        p_load = np.array([b.p_load_min_mw for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_min_mvar for b in net.buses], dtype=float)
    else:
        p_load = np.array([b.p_load_peak_mw * load_scale for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_peak_mvar * load_scale for b in net.buses], dtype=float)
    p_gen = np.zeros(n, dtype=float)
    q_gen = np.zeros(n, dtype=float)
    p_gen[1] = p_new_mw + p_existing_mw
    if reg.uses_cos_phi_p:
        q_gen[1] = float(compute_cos_phi_p_q_mvar(p_new_mw, p_nom_mw=max(p_new_mw, 1e-6)))
    if reg.uses_qu_droop and p_new_mw > 1e-6:
        return solve_ac_powerflow(
            net, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load,
            q_of_v_buses=np.array([1]), q_of_v=make_qu_law(np.array([qu_q_limit_mvar(p_new_mw, p_new_mw)])),
        )
    return solve_ac_powerflow(net, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load)


def find_gpz_substation_hosting_capacity_mw(
    net_n1: NetworkModel,
    regime_code: str,
    p_existing_mw: float = GPZ_EXISTING_GEN_MW,
    v_max_limit_pu: float = V_MAX_PU,
) -> dict[str, float | str]:
    """Find the GPZ 110/15 kV hosting capacity [MW] for new generation at the 15 kV busbar under N-1.

    The network passed in has one 25 MVA transformer out of service. Feasible means: remaining transformer
    S <= 25 MVA, every bus voltage <= v_max_limit_pu and every 15 kV branch within its ampacity, at minimum
    load with the existing busbar generation at full output.
    """
    def _feasible(res: PowerFlowResult) -> tuple[bool, str]:
        if res.tr_loading_n1_pct > 100.0:
            return False, "N-1 transformer (25 MVA)"
        if float(np.max(res.v_mag_pu[1:])) > v_max_limit_pu:
            return False, "Bus voltage (+5.0 %)"
        if float(np.max(res.branch_loading_pct[1:])) > 100.0:
            return False, "Feeder ampacity"
        return True, ""

    lo, hi = 0.0, 50.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        ok, _ = _feasible(solve_gpz_busbar_injection(net_n1, mid, regime_code, p_existing_mw))
        if ok:
            lo = mid
        else:
            hi = mid

    res_opt = solve_gpz_busbar_injection(net_n1, lo, regime_code, p_existing_mw)
    _, binding = _feasible(solve_gpz_busbar_injection(net_n1, hi, regime_code, p_existing_mw))
    return {
        "feeder": "GPZ_N1",
        "regime": regime_code,
        "hosting_capacity_mw": float(np.floor(lo * 100.0) / 100.0),  # rounded down: stays feasible
        "v_pcc_pu": round(float(np.max(res_opt.v_mag_pu[1:])), 4),
        "max_branch_loading_pct": round(float(res_opt.tr_loading_n1_pct), 1),
        "q_net_mvar": round(float(res_opt.q_inj_mvar[1] + net_n1.buses[1].q_load_min_mvar), 2),
        "feeder_loss_kw": round(float(res_opt.total_p_loss_mw * 1000.0), 1),
        "binding_constraint": binding,
    }


def compute_hosting_capacity_table(net_n1: NetworkModel | None = None) -> pd.DataFrame:
    """Compute the hosting capacity matrix across F1, F2, F3 and the GPZ busbar (N-1) for all regimes."""
    net = net_n1 or build_polish_gpz_network(n_transformers=1)
    rows = []
    for fcode in ["F1", "F2", "F3"]:
        for rcode in CONTROL_REGIMES:
            rows.append(find_feeder_hosting_capacity_mw(net, fcode, rcode))
    for rcode in CONTROL_REGIMES:
        rows.append(find_gpz_substation_hosting_capacity_mw(net, rcode))
    return pd.DataFrame(rows)


def compute_annual_reactive_loss_comparison(
    net: NetworkModel | None = None,
    df_profiles: pd.DataFrame | None = None,
    feeder_code: str = "F1",
    p_inst_mw: float = 2.0,
) -> dict[str, float]:
    """Hourly AC power flow over all generating hours: reactive energy and extra feeder losses per regime.

    Extra losses are the feeder I^2 R losses of a regime minus those of the unity power factor run in the
    same hour (signed, summed over the year).
    """
    net = net or build_polish_gpz_network(n_transformers=2)
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    fspec = net.feeders[feeder_code]
    br_ids = list(fspec.branch_ids)
    plant_buses = [fspec.pcc_bus_id, fspec.bus_ids[-2]]

    pv_cf = df["pv_cf"].to_numpy()
    load_sc = df["load_scale"].to_numpy()
    gen_hours = np.flatnonzero(pv_cf > 0.0)

    q_abs = {"cos_phi_p": 0.0, "qu_droop": 0.0}
    hours_q = {"cos_phi_p": 0, "qu_droop": 0}
    extra_loss = {"cos_phi_p": 0.0, "qu_droop": 0.0}
    v_max = {"uncontrolled": 0.0, "cos_phi_p": 0.0, "qu_droop": 0.0}
    for h in gen_hours:
        cf_h = float(pv_cf[h])
        ld_h = float(load_sc[h])
        res = {
            rc: solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, rc, cf_avail=cf_h, load_scale=ld_h)
            for rc in v_max
        }
        loss_unc = float(np.sum(res["uncontrolled"].branch_p_loss_mw[br_ids]))
        for rc, r in res.items():
            v_max[rc] = max(v_max[rc], float(r.v_mag_pu[fspec.pcc_bus_id]))
        q_plant = {
            "cos_phi_p": float(compute_cos_phi_p_q_mvar(p_inst_mw * cf_h, p_nom_mw=p_inst_mw)),
            "qu_droop": float(np.sum(res["qu_droop"].q_v_mvar[plant_buses])),
        }
        for rc in q_abs:
            q_abs[rc] += max(0.0, -q_plant[rc])
            hours_q[rc] += int(q_plant[rc] < -1e-9)
            extra_loss[rc] += float(np.sum(res[rc].branch_p_loss_mw[br_ids])) - loss_unc

    return {
        "p_inst_mw": p_inst_mw,
        "generating_hours": int(len(gen_hours)),
        "energy_mwh": round(float(np.sum(pv_cf)) * p_inst_mw, 1),
        "q_hours_cosphi": hours_q["cos_phi_p"],
        "q_hours_qu": hours_q["qu_droop"],
        "q_absorbed_cosphi_mvarh": round(q_abs["cos_phi_p"], 1),
        "q_absorbed_qu_mvarh": round(q_abs["qu_droop"], 1),
        "extra_loss_cosphi_mwh": round(extra_loss["cos_phi_p"], 3),
        "extra_loss_qu_mwh": round(extra_loss["qu_droop"], 3),
        "q_reduction_pct": round(100.0 * (1.0 - q_abs["qu_droop"] / max(1e-9, q_abs["cos_phi_p"])), 1),
        "extra_loss_reduction_pct": round(
            100.0 * (1.0 - extra_loss["qu_droop"] / max(1e-9, extra_loss["cos_phi_p"])), 1
        ),
        "v_pcc_max_uncontrolled_pu": round(v_max["uncontrolled"], 4),
        "v_pcc_max_cosphi_pu": round(v_max["cos_phi_p"], 4),
        "v_pcc_max_qu_pu": round(v_max["qu_droop"], 4),
    }


def dispatch_export_limited_bess(
    p_avail_mw: np.ndarray,
    p_limit_mw: float,
    p_bess_mw: float,
    e_bess_mwh: float,
    eta_roundtrip: float = 0.90,
    soc0_mwh: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Hourly dispatch of a battery sharing an export limit with the generation behind it.

    Each hour the battery charges from generation above p_limit_mw (limited by its power and free energy),
    and otherwise discharges into the remaining headroom under the limit (limited by power and stored energy).
    Generation that can be neither exported nor stored is spilled. Charge and discharge efficiencies are both
    sqrt(eta_roundtrip).
    """
    p_avail = np.asarray(p_avail_mw, dtype=float)
    eta = float(np.sqrt(eta_roundtrip))
    n = len(p_avail)
    export = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    spilled = np.zeros(n)
    soc = np.zeros(n)
    s = float(soc0_mwh)
    for t in range(n):
        surplus = p_avail[t] - p_limit_mw
        if surplus > 0.0:
            ch = min(surplus, p_bess_mw, max(0.0, e_bess_mwh - s) / eta)
            s += ch * eta
            charge[t] = ch
            spilled[t] = surplus - ch
            export[t] = p_limit_mw
        else:
            dis = min(p_bess_mw, -surplus, s * eta)
            s -= dis / eta
            discharge[t] = dis
            export[t] = p_avail[t] + dis
        soc[t] = s
    return {
        "export_mw": export,
        "charge_mw": charge,
        "discharge_mw": discharge,
        "spilled_mw": spilled,
        "soc_mwh": soc,
        "conversion_loss_mwh": float(np.sum(charge) - np.sum(discharge) - (s - soc0_mwh)),
    }


def compute_curtailment_tradeoff(
    df_profiles: pd.DataFrame | None = None,
    bess_power_share: float = 0.35,
    bess_hours: float = 2.0,
) -> pd.DataFrame:
    """Annual energy cost of an active power export cap (50 %..100 % of P_inst) for PV, with and without BESS.

    Per MW of installed PV (AC). The battery is sized relative to P_inst (bess_power_share * P_inst, bess_hours
    of storage) and dispatched hourly by dispatch_export_limited_bess. installed_per_export_mw = 1 / cap is the
    arithmetic oversizing the cap allows on a fixed export limit; energy_gain_vs_firm_pct is the extra annual
    energy that oversized, capped plant delivers through the same export limit compared with a firm plant.
    """
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    pv_cf = df["pv_cf"].to_numpy()
    total_avail = float(np.sum(pv_cf))

    caps = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    rows = []
    for cap in caps:
        exported = np.minimum(pv_cf, cap)
        bess = dispatch_export_limited_bess(pv_cf, cap, bess_power_share, bess_power_share * bess_hours)
        exported_bess = float(np.sum(bess["export_mw"]))
        rows.append(
            {
                "export_cap_pct": int(round(cap * 100)),
                "export_cap_share": cap,
                "curtailed_energy_pct": round(100.0 * (total_avail - float(np.sum(exported))) / total_avail, 2),
                "curtailed_hours_per_yr": int(np.sum(pv_cf > cap + 1e-9)),
                "bess_lost_energy_pct": round(100.0 * (total_avail - exported_bess) / total_avail, 2),
                "installed_per_export_mw": round(1.0 / cap, 3),
                "annual_export_mwh_per_mw_grid": round(float(np.sum(exported)) / cap, 1),
                "energy_gain_vs_firm_pct": round(100.0 * (float(np.sum(exported)) / cap / total_avail - 1.0), 1),
                "bess_energy_gain_vs_firm_pct": round(100.0 * (exported_bess / cap / total_avail - 1.0), 1),
            }
        )
    return pd.DataFrame(rows)


def compute_export_cap_by_pv_year(cap_pct: int = 70) -> pd.DataFrame:
    """The export-cap trade-off at ``cap_pct`` for every PVGIS year 2005-2023 (sensitivity to the weather year)."""
    rows = []
    for pv_year in sorted(set(load_pvgis_hourly().index.year)):
        dc = pv_dc_year(pv_year)
        pv_cf = np.minimum(PV_DC_AC_RATIO * dc, 1.0)
        row = compute_curtailment_tradeoff(pd.DataFrame({"pv_cf": pv_cf}))
        row = row[row["export_cap_pct"] == cap_pct].iloc[0]
        rows.append({
            "pv_year": pv_year,
            "dc_yield_kwh_per_kwp": round(float(dc.sum()), 1),
            "ac_full_load_hours": round(float(pv_cf.sum()), 1),
            "export_cap_pct": cap_pct,
            "curtailed_energy_pct": float(row["curtailed_energy_pct"]),
            "curtailed_hours_per_yr": int(row["curtailed_hours_per_yr"]),
            "bess_lost_energy_pct": float(row["bess_lost_energy_pct"]),
            "energy_gain_vs_firm_pct": float(row["energy_gain_vs_firm_pct"]),
        })
    return pd.DataFrame(rows)


def compute_n1_transformer_study(
    net_n1: NetworkModel | None = None,
    df_profiles: pd.DataFrame | None = None,
    p_firm_mw: float = 20.0,
    export_cap_share: float = 0.70,
    wind_share: float = 0.50,
    bess_power_share: float = 0.25,
    bess_hours: float = 2.0,
    p_existing_mw: float = GPZ_EXISTING_GEN_MW,
) -> dict:
    """Hourly AC power flow of the GPZ under N-1 (one 25 MVA transformer out) for three busbar scenarios.

    All new generation sits at the GPZ 15 kV busbar at cos phi = 1 with an export limit of p_firm_mw:
      A. firm PV of p_firm_mw;
      B. PV of p_firm_mw / export_cap_share with export capped at p_firm_mw;
      C. cable pooling: PV of p_firm_mw plus wind of wind_share * p_firm_mw behind the same p_firm_mw limit,
         with a co-located battery of bess_power_share * p_firm_mw for bess_hours (dispatch_export_limited_bess).
    Existing busbar PV (p_existing_mw) follows the same PV profile; loads follow load_scale x peak load.
    Each hour is solved with the Newton-Raphson power flow on the N-1 network (OLTC regulating the busbar).
    """
    net = net_n1 or build_polish_gpz_network(n_transformers=1)
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    pv_cf = df["pv_cf"].to_numpy()
    wind_cf = df["wind_cf"].to_numpy()
    load_sc = df["load_scale"].to_numpy()
    n_h = len(df)

    p_a = p_firm_mw * pv_cf
    p_pv_b = p_firm_mw / export_cap_share * pv_cf
    p_b = np.minimum(p_pv_b, p_firm_mw)
    p_avail_c = p_firm_mw * pv_cf + wind_share * p_firm_mw * wind_cf
    bess = dispatch_export_limited_bess(
        p_avail_c, p_firm_mw, bess_power_share * p_firm_mw, bess_power_share * p_firm_mw * bess_hours
    )
    p_c = bess["export_mw"]
    p_c_nobess = np.minimum(p_avail_c, p_firm_mw)

    scen = {"firm_pv": p_a, "capped_pv": p_b, "pooling_bess": p_c}
    s_mva = {k: np.zeros(n_h) for k in scen}
    v_max_pu = {k: 0.0 for k in scen}
    for h in range(n_h):
        solved: dict[float, tuple[float, float]] = {}  # scenarios with identical injection share one solve
        for k, p_new in scen.items():
            key = float(p_new[h])
            if key not in solved:
                res = solve_gpz_busbar_injection(
                    net, key, "uncontrolled", p_existing_mw * float(pv_cf[h]), load_scale=float(load_sc[h])
                )
                solved[key] = (res.tr_s_mva, float(np.max(res.v_mag_pu[1:])))
            s_mva[k][h], v_h = solved[key]
            v_max_pu[k] = max(v_max_pu[k], v_h)

    s_lim = net.substation.s_n1_mva
    df_hm = pd.DataFrame({"month": df["month"].to_numpy(), "hour": df["hour"].to_numpy(), "s_mva": s_mva["pooling_bess"]})
    hm_pivot = df_hm.pivot_table(index="month", columns="hour", values="s_mva", aggfunc=lambda x: np.percentile(x, 95))
    idx_dc = np.linspace(0, n_h - 1, 181, dtype=int)

    out: dict = {
        "p_firm_mw": p_firm_mw,
        "p_existing_mw": p_existing_mw,
        "export_cap_share": export_cap_share,
        "capped_pv_mw": round(p_firm_mw / export_cap_share, 2),
        "pooling_pv_mw": p_firm_mw,
        "pooling_wind_mw": round(wind_share * p_firm_mw, 2),
        "pooling_bess_mw": round(bess_power_share * p_firm_mw, 2),
        "pooling_bess_mwh": round(bess_power_share * p_firm_mw * bess_hours, 2),
        "s_limit_mva": s_lim,
        "duration_hours": [int(x) for x in idx_dc],
        "heatmap_p95_mva": np.round(hm_pivot.to_numpy(), 2).tolist(),
    }
    energy = {"firm_pv": float(np.sum(p_a)), "capped_pv": float(np.sum(p_b)), "pooling_bess": float(np.sum(p_c))}
    for k in scen:
        out[f"s_mva_{k}"] = [round(float(x), 2) for x in np.sort(s_mva[k])[::-1][idx_dc]]
        out[f"max_s_{k}_mva"] = round(float(np.max(s_mva[k])), 2)
        out[f"hours_over_limit_{k}"] = int(np.sum(s_mva[k] > s_lim + 1e-6))
        out[f"hours_above_90pct_{k}"] = int(np.sum(s_mva[k] > 0.9 * s_lim))
        out[f"max_v_{k}_pu"] = round(v_max_pu[k], 4)
        out[f"export_gwh_{k}"] = round(energy[k] / 1000.0, 2)
    out["capped_curtailed_pct"] = round(100.0 * (1.0 - float(np.sum(p_b)) / float(np.sum(p_pv_b))), 2)
    out["pooling_spilled_nobess_pct"] = round(100.0 * (1.0 - float(np.sum(p_c_nobess)) / float(np.sum(p_avail_c))), 2)
    out["pooling_lost_bess_pct"] = round(100.0 * (1.0 - float(np.sum(p_c)) / float(np.sum(p_avail_c))), 2)
    out["pooling_bess_conversion_loss_mwh"] = round(float(bess["conversion_loss_mwh"]), 1)
    return out


def sunniest_week_start(df_profiles: pd.DataFrame, months: tuple[int, ...] = (5, 6)) -> pd.Timestamp:
    """Local midnight starting the 7 consecutive days within ``months`` with the most PV energy."""
    ts_pl = df_profiles["ts_local"]
    day = ts_pl.dt.normalize()
    daily = df_profiles["pv_cf"].groupby(day.to_numpy()).sum()
    daily = daily[daily.index.month.isin(months)]
    week = daily.rolling(7).sum().shift(-6).dropna()
    week = week[(week.index + pd.Timedelta(days=6)).month.isin(months)]
    return week.idxmax()


def simulate_sunny_week_trace(
    net: NetworkModel | None = None,
    df_profiles: pd.DataFrame | None = None,
    feeder_code: str = "F1",
    p_inst_mw: float = 3.0,
) -> dict:
    """Simulate the sunniest May-June week of the profile (168 hours) on a feeder across the control regimes."""
    net = net or build_polish_gpz_network(n_transformers=2)
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    fspec = net.feeders[feeder_code]
    br_ids = list(fspec.branch_ids)

    ts_pl = df["ts_local"]
    start = sunniest_week_start(df)
    mask = (ts_pl >= start) & (ts_pl < start + pd.Timedelta(days=7))
    sub = df.loc[mask]
    last = start + pd.Timedelta(days=6)
    label = (f"{start.day}-{last.day} {last:%b}" if start.month == last.month
             else f"{start.day} {start:%b}-{last.day} {last:%b}")

    timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in sub["ts_local"]]
    pv_cf = sub["pv_cf"].to_numpy()
    load_sc = sub["load_scale"].to_numpy()

    modes = list(CONTROL_REGIMES)
    traces: dict[str, dict[str, list[float]]] = {
        m: {"v_pcc_pu": [], "q_pcc_mvar": [], "p_exp_mw": [], "loss_kw": []} for m in modes
    }
    plant_buses = [fspec.pcc_bus_id, fspec.bus_ids[-2]]

    for k in range(len(sub)):
        cf_k = float(pv_cf[k])
        ld_k = float(load_sc[k])
        for m in modes:
            res = solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, m, cf_avail=cf_k, load_scale=ld_k)
            q_plant = float(
                sum(res.q_inj_mvar[b] + net.buses[b].q_load_peak_mvar * ld_k for b in plant_buses)
            )
            traces[m]["v_pcc_pu"].append(round(float(res.v_mag_pu[fspec.pcc_bus_id]), 4))
            traces[m]["q_pcc_mvar"].append(round(q_plant, 3))
            traces[m]["p_exp_mw"].append(round(p_inst_mw * cf_k, 3))
            traces[m]["loss_kw"].append(round(float(np.sum(res.branch_p_loss_mw[br_ids]) * 1000.0), 1))

    return {
        "timestamps": timestamps,
        "label": label,
        "feeder": feeder_code,
        "p_inst_mw": p_inst_mw,
        "pv_avail_mw": [round(float(p_inst_mw * x), 3) for x in pv_cf],
        "modes": traces,
    }
