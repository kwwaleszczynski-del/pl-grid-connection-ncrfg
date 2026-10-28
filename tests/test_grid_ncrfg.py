"""Test suite for Polish MV/110 kV grid connection study and NC RfG reactive control."""

from __future__ import annotations

import json
import numpy as np
import pytest

from grid_ncrfg.network import (
    BASE_MVA,
    BASE_KV_MV,
    BASE_KV_HV,
    LINE_TYPES,
    build_polish_gpz_network,
    build_Ybus,
)
from grid_ncrfg.powerflow import (
    solve_ac_powerflow,
    solve_two_bus_analytical,
    verify_two_bus_analytical,
)
from grid_ncrfg.ncrfg import (
    CONTROL_REGIMES,
    compute_cos_phi_p_q_mvar,
    compute_qu_droop_q_mvar,
    solve_feeder_with_ncrfg,
)
from grid_ncrfg.profiles import generate_annual_polish_profiles
from grid_ncrfg.hosting import (
    compute_hosting_capacity_table,
    compute_annual_reactive_loss_comparison,
    compute_curtailment_tradeoff,
    compute_n1_transformer_study,
    simulate_sunny_week_trace,
)
from grid_ncrfg.cli import main as cli_main
from grid_ncrfg.pipeline import REPO_ROOT


@pytest.fixture(scope="module")
def hc_table():
    net = build_polish_gpz_network(n_transformers=1)
    return compute_hosting_capacity_table(net)


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


def test_two_bus_multiple_operating_points():
    """Verify 2-bus analytical equation across inductive, capacitive, and load states."""
    for p_pu, q_pu in [(0.45, 0.0), (0.85, -0.28), (-0.30, -0.12), (0.60, 0.15)]:
        v1, th1, ploss, qloss = solve_two_bus_analytical(1.00, 0.075, 0.082, p_pu, q_pu)
        assert 0.85 < v1 < 1.20
        assert -0.3 < th1 < 0.3
        assert ploss >= 0.0
        assert qloss >= 0.0


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


def test_network_topology_and_feeder_rx_ratios():
    """Check the four Polish 15 kV feeders have realistic lengths and R/X ratios."""
    net = build_polish_gpz_network(n_transformers=2)
    assert len(net.feeders) == 4
    f1 = net.feeders["F1"]
    f2 = net.feeders["F2"]
    f3 = net.feeders["F3"]
    f4 = net.feeders["F4"]

    assert f1.total_length_km == pytest.approx(14.2)
    assert f2.total_length_km == pytest.approx(9.6)
    assert f3.total_length_km == pytest.approx(6.4)
    assert f4.total_length_km == pytest.approx(5.1)

    assert 0.88 <= f1.rx_ratio <= 1.05
    assert 0.30 <= f3.rx_ratio <= 0.45
    assert f1.rx_ratio > f3.rx_ratio * 2.0


def test_line_types_thermal_ratings():
    """Standard Polish MV conductor types must have positive impedance and thermal MVA ratings."""
    for code, lt in LINE_TYPES.items():
        assert lt.r_ohm_per_km > 0.0
        assert lt.x_ohm_per_km > 0.0
        assert lt.rx_ratio > 0.0
        assert 8.0 < lt.s_thermal_mva < 16.0


def test_substation_transformer_impedance_scaling():
    """Two parallel 25 MVA transformers in N-0 must have half the series impedance of N-1."""
    net_n0 = build_polish_gpz_network(n_transformers=2)
    net_n1 = build_polish_gpz_network(n_transformers=1)
    assert net_n0.substation.s_n0_mva == pytest.approx(50.0)
    assert net_n1.substation.s_n1_mva == pytest.approx(25.0)
    z0 = net_n0.substation.transformer_z_pu(2)
    z1 = net_n1.substation.transformer_z_pu(1)
    assert z1 == pytest.approx(2.0 * z0)


def test_oltc_tap_step_bounds():
    """OLTC tap changer must clip steps within +-9 x 1.78 %."""
    net_hi = build_polish_gpz_network(n_transformers=2, oltc_tap_step=20)
    net_lo = build_polish_gpz_network(n_transformers=2, oltc_tap_step=-20)
    assert net_hi.oltc_tap_ratio == pytest.approx(1.0 + 9 * 0.0178)
    assert net_lo.oltc_tap_ratio == pytest.approx(1.0 - 9 * 0.0178)


def test_full_network_power_balance_conservation():
    """Active and reactive power conservation across the 20-bus Polish GPZ network to < 1e-9 MW."""
    net = build_polish_gpz_network(n_transformers=2)
    n = len(net.buses)
    p_gen = np.zeros(n)
    q_gen = np.zeros(n)
    p_gen[net.feeders["F1"].pcc_bus_id] = 6.0
    q_gen[net.feeders["F1"].pcc_bus_id] = -1.8
    p_gen[net.feeders["F2"].pcc_bus_id] = 5.0

    p_load = np.array([b.p_load_min_mw for b in net.buses])
    q_load = np.array([b.q_load_min_mvar for b in net.buses])

    res = solve_ac_powerflow(net, p_gen_mw=p_gen, q_gen_mvar=q_gen, p_load_mw=p_load, q_load_mvar=q_load)
    assert res.converged
    assert res.max_mismatch_pu < 1e-11

    p_net_non_slack = float(np.sum(p_gen[1:] - p_load[1:]))
    p_branch_losses = float(np.sum(res.branch_p_loss_mw))
    assert abs(res.p_slack_mw + p_net_non_slack - p_branch_losses) < 1e-9


def test_qu_droop_deadband_and_saturation():
    """NC RfG Q(U) droop must output 0 inside [0.98, 1.02] pu and saturate at +-0.33 Pn."""
    p_nom = 10.0
    for u_in in [0.980, 0.992, 1.000, 1.015, 1.020]:
        assert compute_qu_droop_q_mvar(u_in, p_nom_mw=p_nom, q_max_ratio=0.33) == pytest.approx(0.0, abs=1e-12)

    assert compute_qu_droop_q_mvar(1.035, p_nom_mw=p_nom, q_max_ratio=0.33) == pytest.approx(-1.65, rel=1e-6)
    assert compute_qu_droop_q_mvar(1.050, p_nom_mw=p_nom, q_max_ratio=0.33) == pytest.approx(-3.30, rel=1e-6)
    assert compute_qu_droop_q_mvar(1.075, p_nom_mw=p_nom, q_max_ratio=0.33) == pytest.approx(-3.30, rel=1e-6)
    assert compute_qu_droop_q_mvar(0.950, p_nom_mw=p_nom, q_max_ratio=0.33) == pytest.approx(+3.30, rel=1e-6)


def test_qu_droop_vectorized_and_low_power_gating():
    """Vectorized Q(U) evaluation and low-active-power gating near zero generation."""
    v_vec = np.array([0.96, 1.00, 1.04])
    q_vec = compute_qu_droop_q_mvar(v_vec, p_nom_mw=10.0, q_max_ratio=0.33)
    assert q_vec.shape == (3,)
    assert q_vec[0] > 0.0 and q_vec[1] == 0.0 and q_vec[2] < 0.0
    assert compute_qu_droop_q_mvar(1.04, p_nom_mw=10.0, p_avail_mw=0.0) == pytest.approx(0.0, abs=1e-12)


def test_cos_phi_p_inductive_absorption():
    """Fixed cos-phi(P) control is zero below 20% Pn and inductive above 30% Pn."""
    q_low = compute_cos_phi_p_q_mvar(1.5, p_nom_mw=10.0)
    q_full = compute_cos_phi_p_q_mvar(10.0, p_nom_mw=10.0)
    assert q_low == pytest.approx(0.0, abs=1e-9)
    assert -3.30 < q_full < -2.40


def test_inductive_q_reduces_feeder_pcc_voltage():
    """Inductive reactive absorption must reduce PCC voltage rise on overhead Feeder F1."""
    net = build_polish_gpz_network(n_transformers=2)
    pcc = net.feeders["F1"].pcc_bus_id
    res_unc = solve_feeder_with_ncrfg(net, "F1", 10.0, "uncontrolled")
    res_fix = solve_feeder_with_ncrfg(net, "F1", 10.0, "cos_phi_fixed")
    res_qu = solve_feeder_with_ncrfg(net, "F1", 10.0, "qu_droop")

    assert res_unc.v_mag_pu[pcc] > 1.050
    assert res_fix.v_mag_pu[pcc] < res_unc.v_mag_pu[pcc]
    assert res_qu.v_mag_pu[pcc] < res_fix.v_mag_pu[pcc]
    assert res_qu.v_mag_pu[pcc] <= 1.050


def test_feeder_f1_hosting_capacity_progression(hc_table):
    """Feeder F1 hosting capacity must strictly increase across the 5 connection regimes."""
    f1 = hc_table[hc_table["feeder"] == "F1"].set_index("regime")["hosting_capacity_mw"]

    assert 5.8 <= f1["uncontrolled"] <= 6.6
    assert 9.1 <= f1["cos_phi_fixed"] <= 10.2
    assert 10.8 <= f1["qu_droop"] <= 12.0
    assert 15.5 <= f1["non_firm_70"] <= 17.2
    assert f1["uncontrolled"] < f1["cos_phi_fixed"] < f1["qu_droop"] < f1["non_firm_70"] < f1["cable_pooling_bess"]


def test_feeder_f3_thermal_vs_f1_voltage_bound(hc_table):
    """Suburban cable F3 must have higher firm hosting capacity than rural overhead F1 and be thermally bound."""
    f1_unc = hc_table[(hc_table["feeder"] == "F1") & (hc_table["regime"] == "uncontrolled")].iloc[0]
    f3_unc = hc_table[(hc_table["feeder"] == "F3") & (hc_table["regime"] == "uncontrolled")].iloc[0]
    f3_fix = hc_table[(hc_table["feeder"] == "F3") & (hc_table["regime"] == "cos_phi_fixed")].iloc[0]
    f3_qu = hc_table[(hc_table["feeder"] == "F3") & (hc_table["regime"] == "qu_droop")].iloc[0]

    assert "voltage" in f1_unc["binding_constraint"].lower()
    assert "thermal" in f3_unc["binding_constraint"].lower()
    assert f3_unc["hosting_capacity_mw"] > f1_unc["hosting_capacity_mw"] * 1.8
    assert f3_fix["hosting_capacity_mw"] < f3_qu["hosting_capacity_mw"]


def test_qu_droop_cuts_annual_reactive_losses_vs_fixed_cosphi():
    """NC RfG Q(U) with deadband [0.98, 1.02] pu must cut annual reactive circulation losses by > 60 %."""
    cmp_res = compute_annual_reactive_loss_comparison(feeder_code="F1", p_inst_mw=6.2)
    assert cmp_res["q_absorbed_qu_mvarh"] < cmp_res["q_absorbed_fixed_mvarh"] * 0.45
    assert cmp_res["reactive_loss_qu_mwh"] < cmp_res["reactive_loss_fixed_mwh"] * 0.45
    assert cmp_res["reactive_loss_reduction_pct"] >= 60.0


def test_curtailment_tradeoff_monotonicity():
    """Tighter active export cap must monotonically increase connectable MW and curtailed MWh."""
    df_curt = compute_curtailment_tradeoff()
    curt = df_curt["curtailed_energy_pct"].to_numpy()
    f1_mw = df_curt["f1_connectable_mw"].to_numpy()
    assert np.all(np.diff(curt) <= 0.0)
    assert np.all(np.diff(f1_mw) < 0.0)
    row_70 = df_curt[df_curt["export_cap_pct"] == 70].iloc[0]
    assert 1.5 <= row_70["curtailed_energy_pct"] <= 3.5
    assert row_70["capacity_gain_pct"] == pytest.approx(42.9, abs=0.2)


def test_n1_vs_n0_transformer_loading():
    """Under N-1 (one 25 MVA transformer tripped), GPZ loading percentage is 2x N-0."""
    net_n0 = build_polish_gpz_network(n_transformers=2)
    net_n1 = build_polish_gpz_network(n_transformers=1)
    res0 = solve_feeder_with_ncrfg(net_n0, "F1", 10.0, "uncontrolled")
    res1 = solve_feeder_with_ncrfg(net_n1, "F1", 10.0, "uncontrolled")
    assert res0.tr_loading_n1_pct == pytest.approx(2.0 * res0.tr_loading_n0_pct, rel=1e-6)
    assert res1.tr_s_mva == pytest.approx(res0.tr_s_mva, rel=2e-2)


def test_n1_pooling_bess_respects_25mva_and_low_spillage():
    """36.5 MW PV+Wind + 2h BESS cable pooling keeps GPZ N-1 <= 25 MVA with < 1.0 % spilled energy."""
    n1 = compute_n1_transformer_study()
    assert n1["max_s_pooling_bess_mva"] <= 25.05
    assert n1["pooling_spilled_energy_pct"] < 1.0
    assert len(n1["heatmap_p95_mva"]) == 12
    assert len(n1["heatmap_p95_mva"][0]) == 24


def test_sunny_may_week_simulation():
    """168-hour sunny May week trace on Feeder F1 must show Q(U) clamping midday voltage <= 1.050 pu."""
    wk = simulate_sunny_week_trace(feeder_code="F1", p_inst_mw=11.4)
    assert len(wk["timestamps"]) == 168
    v_unc = np.array(wk["modes"]["uncontrolled"]["v_pcc_pu"])
    v_qu = np.array(wk["modes"]["qu_droop"]["v_pcc_pu"])
    assert np.max(v_unc) > 1.070
    assert np.max(v_qu) <= 1.051


def test_annual_polish_profiles_bounds_and_seasonality():
    """8,760-hour Polish profiles must have valid bounds and realistic summer/winter seasonality."""
    df = generate_annual_polish_profiles()
    assert len(df) == 8760
    assert df["pv_cf"].min() >= 0.0 and df["pv_cf"].max() <= 1.0001
    assert df["wind_cf"].min() >= 0.0 and df["wind_cf"].max() <= 1.0001
    assert df["load_scale"].min() >= 0.30 and df["load_scale"].max() <= 1.0001

    summer_pv = df.loc[df["month"].isin([5, 6, 7]), "pv_cf"].mean()
    winter_pv = df.loc[df["month"].isin([11, 12, 1]), "pv_cf"].mean()
    assert summer_pv > winter_pv * 2.5


def test_cli_verify_command(tmp_path):
    """CLI verify subcommand must write exact verification JSON and return 0."""
    out_file = tmp_path / "ver.json"
    rc = cli_main(["verify", "--out", str(out_file)])
    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["v_abs_error_pu"] < 1e-10


def test_generated_artifacts_and_dashboard_files():
    """Check that pipeline outputs in results/ and docs/ exist and are non-empty."""
    assert (REPO_ROOT / "results" / "hosting_capacity.csv").exists()
    assert (REPO_ROOT / "results" / "curtailment_tradeoff.csv").exists()
    assert (REPO_ROOT / "results" / "verification.json").exists()
    assert (REPO_ROOT / "results" / "summary.json").exists()
    assert (REPO_ROOT / "docs" / ".nojekyll").exists()
    assert (REPO_ROOT / "docs" / "data.js").exists()
    assert (REPO_ROOT / "docs" / "index.html").exists()
    for img_name in [
        "hosting_capacity.png",
        "voltage_qu_profile.png",
        "curtailment_tradeoff.png",
        "n1_transformer_loading.png",
    ]:
        p = REPO_ROOT / "docs" / "img" / img_name
        assert p.exists() and p.stat().st_size > 10_000
