"""Test suite for Polish MV/110 kV grid connection study and NC RfG reactive control.

The tests check physical invariants and internal consistency (solver accuracy, constraint feasibility at the
reported hosting capacity, energy balances, agreement between generated outputs), not target values.
"""

from __future__ import annotations

import csv
import json
import numpy as np
import pytest

from grid_ncrfg.network import (
    BASE_KV_MV,
    LINE_TYPES,
    acsr_ampacity_a,
    build_polish_gpz_network,
    build_Ybus,
)
from grid_ncrfg.powerflow import (
    solve_ac_powerflow,
    solve_two_bus_analytical,
    verify_two_bus_analytical,
)
from grid_ncrfg.ncrfg import (
    QU_Q_MAX_RATIO,
    compute_cos_phi_p_q_mvar,
    compute_qu_droop_q_mvar,
    make_qu_law,
    qu_droop_q_and_slope,
    solve_feeder_with_ncrfg,
)
from grid_ncrfg.profiles import PV_DC_AC_RATIO, PV_YEAR, WIND_FLH, generate_annual_polish_profiles, pv_dc_year
from grid_ncrfg.hosting import (
    compute_annual_reactive_loss_comparison,
    compute_curtailment_tradeoff,
    compute_hosting_capacity_table,
    compute_n1_transformer_study,
    dispatch_export_limited_bess,
    simulate_sunny_week_trace,
    solve_gpz_busbar_injection,
)
from grid_ncrfg.cli import main as cli_main
from grid_ncrfg.pipeline import REPO_ROOT


@pytest.fixture(scope="module")
def net_n1():
    return build_polish_gpz_network(n_transformers=1)


@pytest.fixture(scope="module")
def hc_table(net_n1):
    return compute_hosting_capacity_table(net_n1)


@pytest.fixture(scope="module")
def profiles():
    return generate_annual_polish_profiles()


def _hc(hc_table, feeder, regime):
    return float(hc_table[(hc_table["feeder"] == feeder) & (hc_table["regime"] == regime)]["hosting_capacity_mw"].iloc[0])


# --- Solver -------------------------------------------------------------------------------------------------

def test_two_bus_analytical_exact_agreement():
    """Newton-Raphson Ybus solver must match closed-form 2-bus solution to < 1e-10 pu."""
    ver = verify_two_bus_analytical()
    assert ver["v_abs_error_pu"] < 1e-10
    assert ver["theta_abs_error_rad"] < 1e-10
    assert ver["p_balance_error_mw"] < 1e-9
    assert ver["q_balance_error_mvar"] < 1e-9
    assert ver["p_loss_error_mw"] < 1e-9
    assert ver["q_loss_error_mvar"] < 1e-9
    assert ver["nr_iterations"] <= 8


def test_two_bus_closed_form_satisfies_power_equation():
    """Closed-form V1, theta1 must satisfy S1 = V1 (V1 - V0)* / Z* and give I^2 R losses."""
    for p_pu, q_pu in [(0.45, 0.0), (0.85, -0.28), (-0.30, -0.12), (0.60, 0.15)]:
        r, x = 0.075, 0.082
        v1, th1, ploss, qloss = solve_two_bus_analytical(1.00, r, x, p_pu, q_pu)
        v1c = v1 * np.exp(1j * th1)
        s1 = v1c * np.conj((v1c - 1.0) / complex(r, x))
        assert s1.real == pytest.approx(p_pu, abs=1e-12)
        assert s1.imag == pytest.approx(q_pu, abs=1e-12)
        assert ploss / qloss == pytest.approx(r / x, rel=1e-12)


def test_two_bus_voltage_collapse_raises():
    """Verify 2-bus analytical solver raises ValueError beyond nose point."""
    with pytest.raises(ValueError):
        solve_two_bus_analytical(1.00, 0.50, 0.60, -2.5, -2.0)


def test_ybus_symmetry_and_shunt_row_sums():
    """Ybus with unit tap must be symmetric with row sums equal to nodal shunt susceptance."""
    net = build_polish_gpz_network(n_transformers=2, oltc_tap_step=0)
    Ybus = build_Ybus(net, tap_override=1.0)
    assert np.allclose(Ybus, Ybus.T, atol=1e-14)
    for i in range(len(net.buses)):
        shunt_sum = sum(
            1j * (br.b_pu / 2.0)
            for br in net.branches
            if br.from_bus == i or br.to_bus == i
        )
        assert abs(np.sum(Ybus[i, :]) - shunt_sum) < 1e-12


def test_full_network_power_balance_and_oltc_regulation():
    """Power conservation across the 20-bus network and OLTC holding the 15 kV busbar at its setpoint."""
    net = build_polish_gpz_network(n_transformers=2)
    n = len(net.buses)
    p_gen = np.zeros(n)
    q_gen = np.zeros(n)
    p_gen[net.feeders["F1"].pcc_bus_id] = 2.0
    q_gen[net.feeders["F1"].pcc_bus_id] = -0.6
    p_gen[net.feeders["F2"].pcc_bus_id] = 4.0

    p_load = np.array([b.p_load_min_mw for b in net.buses])
    q_load = np.array([b.q_load_min_mvar for b in net.buses])

    res = solve_ac_powerflow(net, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load)
    assert res.converged
    assert res.max_mismatch_pu < 1e-11
    assert res.v_mag_pu[1] == pytest.approx(net.substation.v_target_mv_pu, abs=1e-5)

    p_net_non_slack = float(np.sum(p_gen[1:] - p_load[1:]))
    q_net_non_slack = float(np.sum(q_gen[1:] - q_load[1:]))
    assert abs(res.p_slack_mw + p_net_non_slack - float(np.sum(res.branch_p_loss_mw))) < 1e-9
    assert abs(res.q_slack_mvar + q_net_non_slack - float(np.sum(res.branch_q_loss_mvar))) < 1e-9


def test_qu_inside_newton_matches_independent_fixed_point():
    """Q(U) solved inside Newton-Raphson must equal a damped fixed-point iteration over plain power flows."""
    net = build_polish_gpz_network(n_transformers=2)
    fs = net.feeders["F1"]
    pcc, adj = fs.pcc_bus_id, fs.bus_ids[-2]
    p_inst = 3.0
    res_nr = solve_feeder_with_ncrfg(net, "F1", p_inst, "qu_droop")
    assert res_nr.converged
    q_law = compute_qu_droop_q_mvar(res_nr.v_mag_pu[[pcc, adj]], p_nom_mw=1.0, q_max_ratio=QU_Q_MAX_RATIO) * p_inst
    assert res_nr.q_v_mvar[pcc] == pytest.approx(0.75 * q_law[0], abs=1e-9)
    assert res_nr.q_v_mvar[adj] == pytest.approx(0.25 * q_law[1], abs=1e-9)

    # Same equilibrium from an independent outer fixed-point loop (fixed tap, so both see the same network)
    n = len(net.buses)
    p_gen = np.zeros(n)
    p_gen[pcc], p_gen[adj] = 0.75 * p_inst, 0.25 * p_inst
    p_load = np.array([b.p_load_min_mw for b in net.buses])
    q_load = np.array([b.q_load_min_mvar for b in net.buses])
    q_lims = QU_Q_MAX_RATIO * p_inst * np.array([0.75, 0.25])
    kw = dict(p_gen_mw=p_gen, p_load_mw=p_load, q_load_mvar=q_load, regulate_gpz_15kv=False)
    res_in = solve_ac_powerflow(net, q_gen_mvar=np.zeros(n), q_of_v_buses=np.array([pcc, adj]),
                                q_of_v=make_qu_law(q_lims), **kw)
    q_gen = np.zeros(n)
    for _ in range(200):
        res_fp = solve_ac_powerflow(net, q_gen_mvar=q_gen, **kw)
        q_t = compute_qu_droop_q_mvar(res_fp.v_mag_pu[[pcc, adj]], p_nom_mw=1.0) * q_lims / QU_Q_MAX_RATIO
        q_gen[[pcc, adj]] = 0.6 * q_gen[[pcc, adj]] + 0.4 * q_t
    assert np.max(np.abs(res_fp.v_mag_pu - res_in.v_mag_pu)) < 1e-9
    assert np.allclose(q_gen[[pcc, adj]], res_in.q_v_mvar[[pcc, adj]], atol=1e-8)


# --- Network data ----------------------------------------------------------------------------------------

def test_network_topology_and_feeder_impedances(net_n1):
    """Feeder lengths and R/X must follow from the branch data; the overhead feeder has the largest |Z|."""
    assert len(net_n1.feeders) == 4
    z_abs = {}
    for code, fs in net_n1.feeders.items():
        br = [net_n1.branches[k] for k in fs.branch_ids]
        assert fs.total_length_km == pytest.approx(sum(b.length_km for b in br))
        assert fs.rx_ratio == pytest.approx(sum(b.r_pu for b in br) / sum(b.x_pu for b in br), abs=1e-3)
        z_abs[code] = abs(complex(sum(b.r_pu for b in br), sum(b.x_pu for b in br)))
    assert z_abs["F1"] == max(z_abs.values())


def test_line_types_physical_ranges():
    """Catalogue-based line data: overhead X > cable X, cable B >> overhead B, MVA rating = sqrt(3) U I."""
    for lt in LINE_TYPES.values():
        assert lt.r_ohm_per_km > 0.0 and lt.x_ohm_per_km > 0.0 and lt.i_dd_a > 0.0
        assert lt.s_thermal_mva == pytest.approx(np.sqrt(3.0) * BASE_KV_MV * lt.i_dd_a / 1000.0)
    overhead = [lt for k, lt in LINE_TYPES.items() if k.startswith("AFL")]
    cables = [lt for k, lt in LINE_TYPES.items() if k.startswith("NA2XS")]
    assert min(o.x_ohm_per_km for o in overhead) > 2.0 * max(c.x_ohm_per_km for c in cables)
    assert min(c.b_us_per_km for c in cables) > 10.0 * max(o.b_us_per_km for o in overhead)


def test_acsr_ampacity_heat_balance_properties():
    """Ampacity grows with conductor size and wind, falls with ambient temperature and solar gain."""
    i_50 = acsr_ampacity_a(9.6, 0.5939)
    i_70 = acsr_ampacity_a(11.3, 0.4324)
    i_120 = acsr_ampacity_a(15.7, 0.2356)
    assert i_50 < i_70 < i_120
    assert acsr_ampacity_a(11.3, 0.4324, wind_m_s=2.0) > i_70
    assert acsr_ampacity_a(11.3, 0.4324, t_amb_c=20.0) > i_70
    assert acsr_ampacity_a(11.3, 0.4324, solar_w_m2=0.0) > i_70
    assert acsr_ampacity_a(11.3, 0.4324, t_amb_c=79.9, solar_w_m2=0.0) < 0.1 * i_70


def test_substation_transformer_impedance_scaling():
    """N-1 network (one of two parallel 25 MVA transformers removed) must have twice the series impedance."""
    net_n0 = build_polish_gpz_network(n_transformers=2)
    net_n1 = build_polish_gpz_network(n_transformers=1)
    assert net_n0.substation.s_n0_mva == pytest.approx(2.0 * net_n1.substation.s_n1_mva)
    z0 = complex(net_n0.branches[0].r_pu, net_n0.branches[0].x_pu)
    z1 = complex(net_n1.branches[0].r_pu, net_n1.branches[0].x_pu)
    assert z1 == pytest.approx(2.0 * z0)
    assert net_n1.branches[0].s_max_mva == pytest.approx(net_n1.substation.s_n1_mva)


def test_oltc_tap_step_bounds():
    """OLTC tap changer must clip steps within +-9 x 1.78 %."""
    net_hi = build_polish_gpz_network(n_transformers=2, oltc_tap_step=20)
    net_lo = build_polish_gpz_network(n_transformers=2, oltc_tap_step=-20)
    assert net_hi.oltc_tap_ratio == pytest.approx(1.0 + 9 * 0.0178)
    assert net_lo.oltc_tap_ratio == pytest.approx(1.0 - 9 * 0.0178)


# --- Control laws ----------------------------------------------------------------------------------------

def test_qu_droop_deadband_and_saturation():
    """Q(U) must output 0 inside [0.98, 1.02] pu, be linear outside and saturate at +-Qmax."""
    p_nom = 10.0
    q_max = QU_Q_MAX_RATIO * p_nom
    for u_in in [0.980, 0.992, 1.000, 1.015, 1.020]:
        assert compute_qu_droop_q_mvar(u_in, p_nom_mw=p_nom) == pytest.approx(0.0, abs=1e-12)
    assert compute_qu_droop_q_mvar(1.035, p_nom_mw=p_nom) == pytest.approx(-0.5 * q_max, rel=1e-9)
    assert compute_qu_droop_q_mvar(1.050, p_nom_mw=p_nom) == pytest.approx(-q_max, rel=1e-9)
    assert compute_qu_droop_q_mvar(1.075, p_nom_mw=p_nom) == pytest.approx(-q_max, rel=1e-9)
    assert compute_qu_droop_q_mvar(0.950, p_nom_mw=p_nom) == pytest.approx(+q_max, rel=1e-9)


def test_qu_slope_matches_numerical_derivative_and_low_power_gating():
    """Analytical dQ/dU used in the Jacobian must equal the numerical derivative; gating below 10 % Pn."""
    v = np.array([0.93, 0.96, 0.97, 1.00, 1.03, 1.04, 1.07])
    q, dq = qu_droop_q_and_slope(v, 2.0)
    h = 1e-7
    q_hi, _ = qu_droop_q_and_slope(v + h, 2.0)
    q_lo, _ = qu_droop_q_and_slope(v - h, 2.0)
    assert np.allclose(dq, (q_hi - q_lo) / (2 * h), atol=1e-5)
    assert compute_qu_droop_q_mvar(1.04, p_nom_mw=10.0, p_avail_mw=0.0) == pytest.approx(0.0, abs=1e-12)
    full = compute_qu_droop_q_mvar(1.04, p_nom_mw=10.0, p_avail_mw=5.0)
    assert compute_qu_droop_q_mvar(1.04, p_nom_mw=10.0, p_avail_mw=0.5) == pytest.approx(0.5 * full)


def test_cos_phi_p_characteristic():
    """cos-phi(P): zero up to the 50 % knee, cos phi = 0.95 inductive at Pn, monotone in between."""
    p = np.linspace(0.0, 10.0, 101)
    q = compute_cos_phi_p_q_mvar(p, p_nom_mw=10.0)
    assert np.all(q[p <= 5.0] == 0.0)
    assert q[-1] == pytest.approx(-10.0 * np.tan(np.arccos(0.95)), rel=1e-12)
    assert np.all(np.diff(q) <= 1e-12)


def test_inductive_q_reduces_feeder_pcc_voltage(hc_table, net_n1):
    """Above the unity-power-factor hosting capacity of F1, both reactive modes must pull the PCC voltage down."""
    pcc = net_n1.feeders["F1"].pcc_bus_id
    p = 1.2 * _hc(hc_table, "F1", "uncontrolled")
    v = {rc: solve_feeder_with_ncrfg(net_n1, "F1", p, rc).v_mag_pu[pcc] for rc in ["uncontrolled", "cos_phi_p", "qu_droop"]}
    assert v["uncontrolled"] > 1.050
    assert v["cos_phi_p"] < v["uncontrolled"]
    assert v["qu_droop"] < v["uncontrolled"]


# --- Hosting capacity ------------------------------------------------------------------------------------

def test_hosting_capacity_is_feasible_and_tight(hc_table, net_n1):
    """At the reported capacity all limits hold; 50 kW more violates the reported binding constraint."""
    for _, row in hc_table.iterrows():
        hc = float(row["hosting_capacity_mw"])
        if row["feeder"] == "GPZ_N1":
            ok = solve_gpz_busbar_injection(net_n1, hc, row["regime"])
            over = solve_gpz_busbar_injection(net_n1, hc + 0.05, row["regime"])
            assert ok.tr_loading_n1_pct <= 100.0 + 1e-6
            assert over.tr_loading_n1_pct > 100.0
            continue
        fs = net_n1.feeders[row["feeder"]]
        ok = solve_feeder_with_ncrfg(net_n1, row["feeder"], hc, row["regime"])
        over = solve_feeder_with_ncrfg(net_n1, row["feeder"], hc + 0.05, row["regime"])
        assert np.max(ok.v_mag_pu[list(fs.bus_ids)]) <= 1.050 + 1e-9
        assert np.max(ok.branch_loading_pct[list(fs.branch_ids)]) <= 100.0 + 1e-6
        v_over = np.max(over.v_mag_pu[list(fs.bus_ids)]) > 1.050
        i_over = np.max(over.branch_loading_pct[list(fs.branch_ids)]) > 100.0
        assert v_over or i_over
        assert ("voltage" in row["binding_constraint"].lower()) == v_over


def test_overhead_feeder_voltage_bound_cable_feeder_thermal_bound(hc_table):
    """Long overhead F1 is voltage-limited and gains from reactive control; short cable F3 is thermally limited."""
    f1 = hc_table[hc_table["feeder"] == "F1"].set_index("regime")
    f3 = hc_table[hc_table["feeder"] == "F3"].set_index("regime")
    assert all("voltage" in s.lower() for s in f1["binding_constraint"])
    assert all("thermal" in s.lower() for s in f3["binding_constraint"])
    assert f1.loc["qu_droop", "hosting_capacity_mw"] > f1.loc["uncontrolled", "hosting_capacity_mw"]
    assert f1.loc["cos_phi_p", "hosting_capacity_mw"] > f1.loc["uncontrolled", "hosting_capacity_mw"]
    # On a thermally bound cable, reactive current competes with active current for ampacity
    assert f3.loc["cos_phi_p", "hosting_capacity_mw"] < f3.loc["uncontrolled", "hosting_capacity_mw"]


def test_gpz_busbar_reactive_absorption_costs_transformer_mva(hc_table):
    """At the OLTC-regulated busbar Q(U) stays idle; cos-phi(P) absorption uses transformer MVA."""
    assert _hc(hc_table, "GPZ_N1", "qu_droop") == pytest.approx(_hc(hc_table, "GPZ_N1", "uncontrolled"), abs=0.01)
    assert _hc(hc_table, "GPZ_N1", "cos_phi_p") < _hc(hc_table, "GPZ_N1", "uncontrolled")


def test_n1_vs_n0_transformer_loading():
    """The same injections load the single remaining transformer twice as much in percent."""
    net_n0 = build_polish_gpz_network(n_transformers=2)
    net_n1 = build_polish_gpz_network(n_transformers=1)
    res0 = solve_gpz_busbar_injection(net_n0, 15.0)
    res1 = solve_gpz_busbar_injection(net_n1, 15.0)
    assert res0.tr_loading_n1_pct == pytest.approx(2.0 * res0.tr_loading_n0_pct, rel=1e-9)
    assert res1.tr_s_mva == pytest.approx(res0.tr_s_mva, rel=2e-2)
    assert res1.tr_s_mva > res0.tr_s_mva  # higher series impedance, higher reactive loss


# --- Time series -----------------------------------------------------------------------------------------

def test_bess_dispatch_invariants():
    """Battery dispatch: export limit, power and energy bounds, and the energy balance must hold."""
    rng = np.random.default_rng(7)
    p_avail = np.clip(rng.normal(0.6, 0.35, size=500), 0.0, 1.3)
    limit, p_b, e_b = 0.8, 0.3, 0.6
    d = dispatch_export_limited_bess(p_avail, limit, p_b, e_b, eta_roundtrip=0.9)
    assert np.all(d["export_mw"] <= limit + 1e-12)
    assert np.all(d["soc_mwh"] >= -1e-12) and np.all(d["soc_mwh"] <= e_b + 1e-12)
    assert np.all(d["charge_mw"] <= p_b + 1e-12) and np.all(d["discharge_mw"] <= p_b + 1e-12)
    assert np.all((d["charge_mw"] == 0.0) | (d["discharge_mw"] == 0.0))
    balance = np.sum(p_avail) - np.sum(d["export_mw"]) - np.sum(d["spilled_mw"]) - d["conversion_loss_mwh"] - d["soc_mwh"][-1]
    assert abs(balance) < 1e-9
    assert d["conversion_loss_mwh"] >= 0.0
    d0 = dispatch_export_limited_bess(p_avail, limit, 0.0, 0.0)
    assert np.allclose(d0["export_mw"], np.minimum(p_avail, limit))


def test_curtailment_tradeoff_properties(profiles):
    """Tighter cap: more curtailment; battery never loses more than plain curtailment; gains bounded by 1/cap."""
    df = compute_curtailment_tradeoff(profiles)
    curt = df["curtailed_energy_pct"].to_numpy()
    assert np.all(np.diff(curt) <= 0.0)
    assert df.loc[df["export_cap_pct"] == 100, "curtailed_energy_pct"].iloc[0] == 0.0
    assert np.all(df["bess_lost_energy_pct"] <= df["curtailed_energy_pct"] + 1e-9)
    gain = df["energy_gain_vs_firm_pct"].to_numpy()
    assert np.all(gain >= 0.0)
    assert np.all(gain <= 100.0 * (df["installed_per_export_mw"].to_numpy() - 1.0) + 0.05)


def test_n1_time_series_short_window(profiles):
    """Hourly N-1 power flow over three May days: sorted duration curves, capped plant exports more energy."""
    window = profiles.iloc[24 * 138: 24 * 141]
    n1 = compute_n1_transformer_study(df_profiles=window, p_firm_mw=18.0)
    for key in ["s_mva_firm_pv", "s_mva_capped_pv", "s_mva_pooling_bess"]:
        curve = np.asarray(n1[key])
        assert np.all(np.diff(curve) <= 1e-12) and np.all(curve >= 0.0)
    assert n1["export_gwh_capped_pv"] >= n1["export_gwh_firm_pv"]
    assert n1["max_s_capped_pv_mva"] >= n1["max_s_firm_pv_mva"] - 1e-9
    assert n1["pooling_lost_bess_pct"] <= n1["pooling_spilled_nobess_pct"] + 1e-9
    assert len(n1["heatmap_p95_mva"][0]) == 24


def test_reactive_comparison_and_week_trace_respect_control_laws(profiles):
    """Q(U) absorbs nothing while the PCC is inside the deadband; cos-phi(P) nothing below its knee."""
    wk = simulate_sunny_week_trace(df_profiles=profiles, feeder_code="F1", p_inst_mw=3.0)
    assert len(wk["timestamps"]) == 168
    v_qu = np.array(wk["modes"]["qu_droop"]["v_pcc_pu"])
    q_qu = np.array(wk["modes"]["qu_droop"]["q_pcc_mvar"])
    in_db = (v_qu > 0.9805) & (v_qu < 1.0195)
    assert in_db.sum() > 50
    assert np.all(np.abs(q_qu[in_db]) < 1e-3)
    q_cos = np.array(wk["modes"]["cos_phi_p"]["q_pcc_mvar"])
    p_exp = np.array(wk["modes"]["cos_phi_p"]["p_exp_mw"])
    assert np.all(np.abs(q_cos[p_exp <= 0.5 * 3.0]) < 1e-3)

    window = profiles.iloc[24 * 138: 24 * 141]
    cmp_res = compute_annual_reactive_loss_comparison(df_profiles=window, feeder_code="F1", p_inst_mw=2.0)
    assert cmp_res["q_absorbed_qu_mvarh"] >= 0.0 and cmp_res["q_absorbed_cosphi_mvarh"] >= 0.0
    assert cmp_res["q_hours_qu"] <= cmp_res["generating_hours"]


def test_annual_profiles_match_calibration(profiles):
    """Profiles must hit their calibration totals and stay within physical bounds."""
    assert len(profiles) == 8760
    assert profiles["wind_cf"].sum() == pytest.approx(WIND_FLH, rel=1e-6)
    dc_yield = pv_dc_year(PV_YEAR).sum()
    assert 900.0 < dc_yield < 1250.0  # kWh/kWp, central Poland
    pv_sum = profiles["pv_cf"].sum()
    assert dc_yield < pv_sum <= PV_DC_AC_RATIO * dc_yield + 1e-6
    assert np.allclose(profiles["pv_cf"], np.minimum(PV_DC_AC_RATIO * pv_dc_year(PV_YEAR), 1.0))
    for col in ["pv_cf", "wind_cf", "load_scale"]:
        assert profiles[col].min() >= 0.0 and profiles[col].max() <= 1.0 + 1e-12
    summer_pv = profiles.loc[profiles["month"].isin([5, 6, 7]), "pv_cf"].mean()
    winter_pv = profiles.loc[profiles["month"].isin([11, 12, 1]), "pv_cf"].mean()
    assert summer_pv > winter_pv * 2.5
    night = profiles["hour"].isin([0, 1, 2, 22, 23])
    assert profiles.loc[night, "pv_cf"].max() == 0.0


def test_cli_verify_command(tmp_path):
    """CLI verify subcommand must write the verification JSON and return 0."""
    out_file = tmp_path / "ver.json"
    rc = cli_main(["verify", "--out", str(out_file)])
    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["v_abs_error_pu"] < 1e-10


# --- Generated outputs -----------------------------------------------------------------------------------

def test_generated_outputs_agree_with_each_other():
    """results/*.csv, results/*.json, docs/data.js and the README must carry the same numbers."""
    summary = json.loads((REPO_ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    data_js = (REPO_ROOT / "docs" / "data.js").read_text(encoding="utf-8")
    assert json.loads(data_js[len("window.DATA = "):].rstrip().rstrip(";")) == summary
    ver = json.loads((REPO_ROOT / "results" / "verification.json").read_text(encoding="utf-8"))
    assert ver == summary["verification"]

    with open(REPO_ROOT / "results" / "hosting_capacity.csv", encoding="utf-8") as fh:
        hc_rows = list(csv.DictReader(fh))
    assert [float(r["hosting_capacity_mw"]) for r in hc_rows] == [r["hosting_capacity_mw"] for r in summary["hosting_table"]]
    with open(REPO_ROOT / "results" / "curtailment_tradeoff.csv", encoding="utf-8") as fh:
        curt_rows = list(csv.DictReader(fh))
    assert [float(r["curtailed_energy_pct"]) for r in curt_rows] == [r["curtailed_energy_pct"] for r in summary["curtailment"]]

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for r in summary["hosting_table"]:
        assert f"{r['hosting_capacity_mw']:.2f} MW" in readme
    hl = summary["headline"]
    assert f"{hl['curt_hl_pct']:.2f} %" in readme
    assert f"{hl['q_reduction_pct']:.1f} %" in readme
    assert f"{ver['p_loss_exact_mw']:.10f}" in readme

    for img_name in ["hosting_capacity.png", "voltage_qu_profile.png", "curtailment_tradeoff.png", "n1_transformer_loading.png"]:
        p = REPO_ROOT / "docs" / "img" / img_name
        assert p.exists() and p.stat().st_size > 10_000
