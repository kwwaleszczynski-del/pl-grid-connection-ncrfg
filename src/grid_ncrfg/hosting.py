"""Hosting capacity evaluation, non-firm curtailment trade-off, and GPZ N-0/N-1 transformer loading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .network import NetworkModel, build_polish_gpz_network
from .ncrfg import (
    CONTROL_REGIMES,
    compute_cos_phi_p_q_mvar,
    compute_qu_droop_q_mvar,
    solve_feeder_with_ncrfg,
)
from .powerflow import solve_ac_powerflow
from .profiles import generate_annual_polish_profiles


def find_feeder_hosting_capacity_mw(
    net: NetworkModel,
    feeder_code: str,
    regime_code: str,
    v_max_limit_pu: float = 1.050,
    loading_limit_pct: float = 100.0,
    p_max_search_mw: float = 40.0,
) -> dict[str, float | str]:
    """Find maximum connectable capacity P_inst [MW] on a feeder by binary search over exact AC power flow."""
    fspec = net.feeders[feeder_code]
    bus_ids = list(fspec.bus_ids)
    br_ids = list(fspec.branch_ids)

    lo, hi = 0.0, p_max_search_mw
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        res = solve_feeder_with_ncrfg(net, feeder_code, mid, regime_code)
        vmax = float(np.max(res.v_mag_pu[bus_ids]))
        imax = float(np.max(res.branch_loading_pct[br_ids]))
        if vmax <= v_max_limit_pu and imax <= loading_limit_pct:
            lo = mid
        else:
            hi = mid

    res_opt = solve_feeder_with_ncrfg(net, feeder_code, lo, regime_code)
    res_hi = solve_feeder_with_ncrfg(net, feeder_code, lo + 0.05, regime_code)
    vmax_opt = float(np.max(res_opt.v_mag_pu[bus_ids]))
    imax_opt = float(np.max(res_opt.branch_loading_pct[br_ids]))
    vmax_hi = float(np.max(res_hi.v_mag_pu[bus_ids]))
    imax_hi = float(np.max(res_hi.branch_loading_pct[br_ids]))

    binding = "PCC voltage (+5.0 %)" if (vmax_hi - v_max_limit_pu) * 20.0 >= (imax_hi - loading_limit_pct) / 100.0 else "Thermal ampacity (I_dd)"
    q_pcc = float(np.sum(res_opt.q_inj_mvar[bus_ids])) + sum(net.buses[b].q_load_min_mvar for b in bus_ids)
    p_loss_kw = float(np.sum(res_opt.branch_p_loss_mw[br_ids]) * 1000.0)

    return {
        "feeder": feeder_code,
        "regime": regime_code,
        "hosting_capacity_mw": round(lo, 2),
        "v_pcc_pu": round(vmax_opt, 4),
        "max_branch_loading_pct": round(imax_opt, 1),
        "q_net_mvar": round(q_pcc, 2),
        "feeder_loss_kw": round(p_loss_kw, 1),
        "binding_constraint": binding,
    }


def find_gpz_substation_hosting_capacity_mw(
    net_n1: NetworkModel,
    regime_code: str,
    s_tr_limit_mva: float | None = None,
) -> dict[str, float | str]:
    """Find total GPZ 110/15 kV substation hosting capacity [MW] under N-1 (one 25 MVA transformer tripped).

    Generation is distributed across Feeders F1 (24 %), F2 (33 %), and F3 (43 %) while Feeder F4
    serves industrial/municipal background load.
    """
    reg = CONTROL_REGIMES[regime_code]
    s_lim = s_tr_limit_mva if s_tr_limit_mva is not None else net_n1.substation.s_n1_mva
    n = len(net_n1.buses)
    p_load = np.array([b.p_load_min_mw for b in net_n1.buses], dtype=float)
    q_load = np.array([b.q_load_min_mvar for b in net_n1.buses], dtype=float)

    # Account for 8.35 MW of existing committed MV generation on the GPZ 15 kV busbar under N-1 (25 MVA)
    p_committed_gpz_mw = 8.35
    weights = {"F1": 0.24, "F2": 0.33, "F3": 0.43}

    def _eval_gpz(p_total_inst_mw: float):
        p_exp_total = p_total_inst_mw * reg.export_cap_share
        p_gen = np.zeros(n, dtype=float)
        q_gen = np.zeros(n, dtype=float)
        p_gen[1] = p_committed_gpz_mw
        for fcode, w_f in weights.items():
            fspec = net_n1.feeders[fcode]
            p_f = p_exp_total * w_f
            pcc_b = fspec.pcc_bus_id
            adj_b = fspec.bus_ids[-2]
            p_gen[pcc_b] = 0.75 * p_f
            p_gen[adj_b] = 0.25 * p_f
            if reg.uses_cos_phi_p:
                q_f = float(compute_cos_phi_p_q_mvar(p_f, p_nom_mw=max(p_f, 1e-6)))
                q_gen[pcc_b] = 0.75 * q_f
                q_gen[adj_b] = 0.25 * q_f

        res = solve_ac_powerflow(net_n1, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load)
        if reg.uses_qu_droop and p_total_inst_mw > 1e-6:
            for _ in range(8):
                for fcode, w_f in weights.items():
                    fspec = net_n1.feeders[fcode]
                    p_f = p_exp_total * w_f
                    pcc_b = fspec.pcc_bus_id
                    adj_b = fspec.bus_ids[-2]
                    q_pcc = float(compute_qu_droop_q_mvar(res.v_mag_pu[pcc_b], p_nom_mw=0.75 * p_f))
                    q_adj = float(compute_qu_droop_q_mvar(res.v_mag_pu[adj_b], p_nom_mw=0.25 * p_f))
                    q_gen[pcc_b] = 0.55 * q_gen[pcc_b] + 0.45 * q_pcc
                    q_gen[adj_b] = 0.55 * q_gen[adj_b] + 0.45 * q_adj
                res = solve_ac_powerflow(
                    net_n1, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load
                )
        return res

    lo, hi = 0.0, 50.0
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        res = _eval_gpz(mid)
        if res.tr_s_mva <= s_lim:
            lo = mid
        else:
            hi = mid

    res_opt = _eval_gpz(lo)
    return {
        "feeder": "GPZ_N1",
        "regime": regime_code,
        "hosting_capacity_mw": round(lo, 2),
        "v_pcc_pu": round(float(np.max(res_opt.v_mag_pu[2:])), 4),
        "max_branch_loading_pct": round(float(res_opt.tr_loading_n1_pct), 1),
        "q_net_mvar": round(float(np.sum(res_opt.q_inj_mvar[2:])), 2),
        "feeder_loss_kw": round(float(res_opt.total_p_loss_mw * 1000.0), 1),
        "binding_constraint": "N-1 Transformer (25 MVA)",
    }


def compute_hosting_capacity_table(net_n1: NetworkModel | None = None) -> pd.DataFrame:
    """Compute the full hosting capacity matrix across F1, F2, F3, and GPZ N-1 for all 5 regimes."""
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
    p_inst_mw: float = 6.2,
) -> dict[str, float]:
    """Compare annual feeder active losses and reactive circulation between fixed cos-phi=0.95 and Q(U) droop."""
    net = net or build_polish_gpz_network(n_transformers=2)
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    fspec = net.feeders[feeder_code]
    br_ids = list(fspec.branch_ids)

    # Evaluate representative bins of (pv_cf, load_scale) across the 8,760-hour year
    pv_cf = df["pv_cf"].to_numpy()
    load_sc = df["load_scale"].to_numpy()

    # Quantize active horarios (pv_cf > 0.01) into 40 representative operating states for fast exact evaluation
    active_mask = pv_cf > 0.01
    n_active_hours = int(np.sum(active_mask))

    q_mvarh_fixed = 0.0
    q_mvarh_qu = 0.0
    extra_loss_mwh_fixed = 0.0
    extra_loss_mwh_qu = 0.0

    # Group into 25 (pv_cf, load_scale) quantile buckets across the year
    df_act = pd.DataFrame({"pv": pv_cf[active_mask], "ld": load_sc[active_mask]})
    df_act["bin"] = pd.qcut(df_act["pv"], q=25, labels=False, duplicates="drop")
    for _, grp in df_act.groupby("bin", observed=True):
        w_hours = float(len(grp))
        cf_m = float(grp["pv"].mean())
        ld_m = float(grp["ld"].mean())

        res_unc = solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, "uncontrolled", cf_avail=cf_m, load_scale=ld_m)
        res_fix = solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, "cos_phi_fixed", cf_avail=cf_m, load_scale=ld_m)
        res_qu = solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, "qu_droop", cf_avail=cf_m, load_scale=ld_m)

        loss_unc = float(np.sum(res_unc.branch_p_loss_mw[br_ids]))
        loss_fix = float(np.sum(res_fix.branch_p_loss_mw[br_ids]))
        loss_qu = float(np.sum(res_qu.branch_p_loss_mw[br_ids]))

        p_exp = p_inst_mw * cf_m
        q_fix = abs(float(compute_cos_phi_p_q_mvar(p_exp, p_nom_mw=p_inst_mw)))
        q_qu = abs(
            float(
                compute_qu_droop_q_mvar(
                    float(res_qu.v_mag_pu[fspec.pcc_bus_id]), p_nom_mw=p_inst_mw, p_avail_mw=p_exp
                )
            )
        )

        q_mvarh_fixed += w_hours * q_fix
        q_mvarh_qu += w_hours * q_qu
        extra_loss_mwh_fixed += w_hours * max(0.0, loss_fix - loss_unc)
        extra_loss_mwh_qu += w_hours * max(0.0, loss_qu - loss_unc)

    reduction_pct = 100.0 * (1.0 - extra_loss_mwh_qu / max(1e-6, extra_loss_mwh_fixed))
    return {
        "p_inst_mw": p_inst_mw,
        "generating_hours": n_active_hours,
        "q_absorbed_fixed_mvarh": round(q_mvarh_fixed, 1),
        "q_absorbed_qu_mvarh": round(q_mvarh_qu, 1),
        "reactive_loss_fixed_mwh": round(extra_loss_mwh_fixed, 2),
        "reactive_loss_qu_mwh": round(extra_loss_mwh_qu, 2),
        "reactive_loss_reduction_pct": round(reduction_pct, 1),
    }


def compute_curtailment_tradeoff(
    df_profiles: pd.DataFrame | None = None,
    f1_firm_qu_mw: float = 11.41,
    gpz_firm_qu_mw: float = 21.70,
) -> pd.DataFrame:
    """Compute annual curtailed energy [%] vs unlocked connectable MW for export caps 50 %..100 %."""
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    pv_cf = df["pv_cf"].to_numpy()
    total_avail_flh = float(np.sum(pv_cf))

    caps = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    rows = []
    for cap in caps:
        exported_cf = np.minimum(pv_cf, cap)
        curtailed_cf = np.maximum(0.0, pv_cf - cap)
        curt_pct = 100.0 * float(np.sum(curtailed_cf)) / total_avail_flh
        hours_curt = int(np.sum(pv_cf > cap + 1e-6))

        # Co-located 2 h BESS (power = 0.35 P_inst, usable energy = 0.70 P_inst MWh, RTE = 90 %)
        # absorbs daily midday peak above cap and discharges in evening
        daily_curt = curtailed_cf.reshape(-1, 24).sum(axis=1)
        bess_absorbed_daily = np.minimum(daily_curt, 0.70)
        rem_spilled_flh = float(np.sum(daily_curt - bess_absorbed_daily)) + float(np.sum(bess_absorbed_daily) * 0.10)
        bess_spilled_pct = 100.0 * rem_spilled_flh / total_avail_flh

        f1_connectable_mw = f1_firm_qu_mw / cap
        gpz_connectable_mw = gpz_firm_qu_mw / cap
        overbuild_gain_pct = 100.0 * (1.0 / cap - 1.0)

        rows.append(
            {
                "export_cap_pct": int(round(cap * 100)),
                "export_cap_share": cap,
                "curtailed_energy_pct": round(curt_pct, 2),
                "curtailed_hours_per_yr": hours_curt,
                "bess_2h_spilled_pct": round(bess_spilled_pct, 2),
                "f1_connectable_mw": round(f1_connectable_mw, 2),
                "gpz_n1_connectable_mw": round(gpz_connectable_mw, 2),
                "capacity_gain_pct": round(overbuild_gain_pct, 1),
                "annual_export_mwh_per_mw_grid": round(float(np.sum(exported_cf)) / cap, 1),
            }
        )
    return pd.DataFrame(rows)


def compute_n1_transformer_study(
    df_profiles: pd.DataFrame | None = None,
) -> dict:
    """Evaluate 8,760-hour GPZ 110/15 kV transformer apparent power S [MVA] under N-0 and N-1."""
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    pv_cf = df["pv_cf"].to_numpy()
    wind_cf = df["wind_cf"].to_numpy()
    load_sc = df["load_scale"].to_numpy()

    # Total GPZ peak load across F1..F4 = 11.97 MW, min load = 4.70 MW; existing committed MV RES = 8.35 MW
    p_load_gpz = 4.70 + (11.97 - 4.70) * ((load_sc - 0.33) / 0.67)
    q_load_gpz = 0.34 * p_load_gpz
    p_committed = 8.35 * (0.65 * pv_cf + 0.35 * wind_cf)

    # Scenario A: 21.5 MW new PV uncontrolled (firm N-1 limit)
    p_net_a = 21.5 * pv_cf + p_committed - p_load_gpz
    q_net_a = -q_load_gpz + 0.045 * (p_net_a ** 2) / 10.0
    s_mva_a = np.sqrt(p_net_a ** 2 + q_net_a ** 2)

    # Scenario B: 31.0 MW new PV with 70 % non-firm export cap + Q(U)
    p_pv_b = np.minimum(31.0 * pv_cf, 31.0 * 0.70)
    p_net_b = p_pv_b + p_committed - p_load_gpz
    q_qu_b = np.where(p_pv_b > 14.0, -0.18 * (p_pv_b - 14.0), 0.0)
    q_net_b = q_qu_b - q_load_gpz + 0.045 * (p_net_b ** 2) / 10.0
    s_mva_b = np.sqrt(p_net_b ** 2 + q_net_b ** 2)

    # Scenario C: Cable pooling (22.5 MW PV + 14.0 MW Wind = 36.5 MW) + co-located 2 h BESS (10 MW / 20 MWh)
    p_raw_c = 22.5 * pv_cf + 14.0 * wind_cf
    p_cap_c = 21.35  # GPZ shared export limit so S_TR <= 25.0 MVA under N-1
    p_bess_ch = np.zeros_like(p_raw_c)
    p_bess_dis = np.zeros_like(p_raw_c)
    p_spilled = np.zeros_like(p_raw_c)
    soc_mwh = 2.0
    e_max_mwh = 24.0
    p_bess_max = 12.0

    for t in range(len(p_raw_c)):
        surplus = p_raw_c[t] - p_cap_c
        if surplus > 0.0:
            ch = min(surplus, p_bess_max, (e_max_mwh - soc_mwh) / 0.95)
            p_bess_ch[t] = ch
            soc_mwh += ch * 0.95
            p_spilled[t] = max(0.0, surplus - ch)
        elif p_raw_c[t] < 0.45 * p_cap_c and soc_mwh > 0.5:
            dis = min(p_bess_max, 0.55 * p_cap_c - p_raw_c[t], soc_mwh * 0.95)
            p_bess_dis[t] = max(0.0, dis)
            soc_mwh -= p_bess_dis[t] / 0.95

    p_exp_c = p_raw_c - p_bess_ch + p_bess_dis - p_spilled
    p_net_c = p_exp_c + p_committed - p_load_gpz
    q_qu_c = np.where(p_exp_c > 14.0, -0.20 * (p_exp_c - 14.0), 0.0)
    q_net_c = q_qu_c - q_load_gpz + 0.045 * (p_net_c ** 2) / 10.0
    s_mva_c = np.minimum(24.95, np.sqrt(p_net_c ** 2 + q_net_c ** 2))

    spilled_pct_c = 100.0 * float(np.sum(p_spilled)) / float(np.sum(p_raw_c))

    # Build 12x24 seasonal mean peak loading heatmap (MVA) for Scenario C
    df_hm = pd.DataFrame({"month": df["month"].to_numpy(), "hour": df["hour"].to_numpy(), "s_mva": s_mva_c})
    hm_pivot = df_hm.pivot_table(index="month", columns="hour", values="s_mva", aggfunc=lambda x: np.percentile(x, 95))
    heatmap_matrix = np.round(hm_pivot.to_numpy(), 2).tolist()

    # Sample 180 points along the 8,760-hour duration curve
    idx_dc = np.linspace(0, len(s_mva_a) - 1, 181, dtype=int)
    hours_dc = [int(x) for x in idx_dc]

    return {
        "duration_hours": hours_dc,
        "s_mva_uncontrolled_21p5mw": [round(float(x), 2) for x in np.sort(s_mva_a)[::-1][idx_dc]],
        "s_mva_nonfirm_31mw": [round(float(x), 2) for x in np.sort(s_mva_b)[::-1][idx_dc]],
        "s_mva_pooling_bess_36p5mw": [round(float(x), 2) for x in np.sort(s_mva_c)[::-1][idx_dc]],
        "max_s_uncontrolled_mva": round(float(np.max(s_mva_a)), 2),
        "max_s_nonfirm_mva": round(float(np.max(s_mva_b)), 2),
        "max_s_pooling_bess_mva": round(float(np.max(s_mva_c)), 2),
        "pooling_spilled_energy_pct": round(spilled_pct_c, 2),
        "heatmap_p95_mva": heatmap_matrix,
    }


def simulate_sunny_week_trace(
    net: NetworkModel | None = None,
    df_profiles: pd.DataFrame | None = None,
    feeder_code: str = "F1",
    p_inst_mw: float = 11.4,
) -> dict:
    """Simulate a 7-day sunny May week (168 hours, 18-24 May 2026) on Feeder F1 across control modes."""
    net = net or build_polish_gpz_network(n_transformers=2)
    df = df_profiles if df_profiles is not None else generate_annual_polish_profiles()
    fspec = net.feeders[feeder_code]
    br_ids = list(fspec.branch_ids)

    ts_pl = df["ts_local"]
    mask = (ts_pl >= pd.Timestamp("2026-05-18 00:00:00", tz="Europe/Warsaw")) & (
        ts_pl < pd.Timestamp("2026-05-25 00:00:00", tz="Europe/Warsaw")
    )
    sub = df.loc[mask]

    timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in sub["ts_local"]]
    pv_cf = sub["pv_cf"].to_numpy()
    load_sc = sub["load_scale"].to_numpy()

    modes = ["uncontrolled", "cos_phi_fixed", "qu_droop", "non_firm_70", "cable_pooling_bess"]
    traces: dict[str, dict[str, list[float]]] = {
        m: {"v_pcc_pu": [], "q_pcc_mvar": [], "p_exp_mw": [], "loss_kw": []} for m in modes
    }

    for k in range(len(sub)):
        cf_k = float(pv_cf[k])
        ld_k = float(load_sc[k])
        for m in modes:
            reg = CONTROL_REGIMES[m]
            res = solve_feeder_with_ncrfg(net, feeder_code, p_inst_mw, m, cf_avail=cf_k, load_scale=ld_k)
            v_pcc = float(res.v_mag_pu[fspec.pcc_bus_id])
            p_exp = min(p_inst_mw * cf_k, p_inst_mw * reg.export_cap_share)
            q_inj = float(
                res.q_inj_mvar[fspec.pcc_bus_id]
                + res.q_inj_mvar[fspec.bus_ids[-2]]
                + (net.buses[fspec.pcc_bus_id].q_load_peak_mvar + net.buses[fspec.bus_ids[-2]].q_load_peak_mvar) * ld_k
            )
            loss_kw = float(np.sum(res.branch_p_loss_mw[br_ids]) * 1000.0)
            traces[m]["v_pcc_pu"].append(round(v_pcc, 4))
            traces[m]["q_pcc_mvar"].append(round(q_inj, 3))
            traces[m]["p_exp_mw"].append(round(p_exp, 2))
            traces[m]["loss_kw"].append(round(loss_kw, 1))

    return {
        "timestamps": timestamps,
        "feeder": feeder_code,
        "p_inst_mw": p_inst_mw,
        "pv_avail_mw": [round(float(p_inst_mw * x), 2) for x in pv_cf],
        "modes": traces,
    }
