"""End-to-end simulation pipeline, Matplotlib figure generator, and interactive Plotly dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .network import NetworkModel, build_polish_gpz_network
from .powerflow import verify_two_bus_analytical
from .ncrfg import CONTROL_REGIMES, solve_feeder_with_ncrfg
from .profiles import generate_annual_polish_profiles
from .hosting import (
    compute_hosting_capacity_table,
    compute_annual_reactive_loss_comparison,
    compute_curtailment_tradeoff,
    compute_n1_transformer_study,
    simulate_sunny_week_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def compute_spatial_voltage_profiles(net: NetworkModel) -> dict:
    """Compute distance-voltage profiles U(x) and branch thermal loadings across F1, F2, F3 for interactive slider."""
    capacities = list(range(2, 21))  # 2 MW .. 20 MW
    out: dict[str, dict] = {}
    for fcode in ["F1", "F2", "F3"]:
        fspec = net.feeders[fcode]
        bus_ids = [1] + list(fspec.bus_ids)
        br_ids = list(fspec.branch_ids)
        distances_km = [round(float(net.buses[b].distance_km), 2) for b in bus_ids]
        bus_names = [net.buses[b].name for b in bus_ids]

        regimes_data: dict[str, dict[str, dict]] = {}
        for rcode in CONTROL_REGIMES:
            regimes_data[rcode] = {}
            for p_mw in capacities:
                res = solve_feeder_with_ncrfg(net, fcode, float(p_mw), rcode)
                v_prof = [round(float(res.v_mag_pu[b]), 4) for b in bus_ids]
                max_v = round(float(np.max(res.v_mag_pu[list(fspec.bus_ids)])), 4)
                max_i_pct = round(float(np.max(res.branch_loading_pct[br_ids])), 1)
                loss_kw = round(float(np.sum(res.branch_p_loss_mw[br_ids]) * 1000.0), 1)
                q_pcc = round(
                    float(
                        res.q_inj_mvar[fspec.pcc_bus_id]
                        + res.q_inj_mvar[fspec.bus_ids[-2]]
                        + net.buses[fspec.pcc_bus_id].q_load_min_mvar
                        + net.buses[fspec.bus_ids[-2]].q_load_min_mvar
                    ),
                    2,
                )
                regimes_data[rcode][str(p_mw)] = {
                    "v_pu": v_prof,
                    "v_max_pu": max_v,
                    "max_loading_pct": max_i_pct,
                    "loss_kw": loss_kw,
                    "q_mvar": q_pcc,
                }
        out[fcode] = {
            "name": fspec.name,
            "description": fspec.description,
            "rx_ratio": fspec.rx_ratio,
            "total_length_km": fspec.total_length_km,
            "distances_km": distances_km,
            "bus_names": bus_names,
            "regimes": regimes_data,
        }
    return out


def generate_plots(
    df_hc: pd.DataFrame,
    spatial_profiles: dict,
    week_trace: dict,
    df_curt: pd.DataFrame,
    n1_study: dict,
    img_dir: Path,
) -> None:
    """Generate the 4 README figures in docs/img/*.png using the exact matplotlib style."""
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    BLUE, NAVY, ORANGE, WARM, GREEN, RED, GREY, INK = (
        "#2a78d6",
        "#1b3f8f",
        "#eb6834",
        "#e8a33d",
        "#1baf7a",
        "#c8423b",
        "#898781",
        "#52514e",
    )

    # 1. Hosting capacity across Feeders F1-F3 & GPZ N-1 + spatial voltage profile on Feeder F1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1))
    locs = ["F1", "F2", "F3", "GPZ_N1"]
    loc_labels = [
        "F1 Rural\n(14.2 km)",
        "F2 Mixed\n(9.6 km)",
        "F3 Cable\n(6.4 km)",
        "GPZ 110/15 kV\n(N-1: 25 MVA)",
    ]
    regime_Order = [
        ("uncontrolled", "1. cos phi = 1.0 (firm)", GREY),
        ("cos_phi_fixed", "2. Fixed cos phi = 0.95 ind", WARM),
        ("qu_droop", "3. NC RfG Q(U) droop", BLUE),
        ("non_firm_70", "4. Q(U) + 70 % export cap", GREEN),
        ("cable_pooling_bess", "5. Q(U) + 2 h BESS pooling", NAVY),
    ]
    x = np.arange(len(locs))
    w = 0.155
    for i, (rcode, rlabel, col) in enumerate(regime_Order):
        vals = [
            float(df_hc[(df_hc["feeder"] == loc) & (df_hc["regime"] == rcode)]["hosting_capacity_mw"].iloc[0])
            for loc in locs
        ]
        ax1.bar(x + (i - 2) * w, vals, width=w, label=rlabel, color=col)
    ax1.set_xticks(x, loc_labels, fontsize=8.5)
    ax1.set_ylabel("Connectable capacity [MW]")
    ax1.set_title("Hosting capacity by feeder and connection regime", fontsize=10)
    ax1.set_ylim(0, 44.0)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Right panel: Spatial voltage profile along Feeder F1 (0..14.2 km) at 11 MW PV
    f1_sp = spatial_profiles["F1"]
    d_km = f1_sp["distances_km"]
    p_ref = "11"
    ax2.plot(
        d_km,
        f1_sp["regimes"]["uncontrolled"][p_ref]["v_pu"],
        marker="o",
        ms=4,
        lw=1.3,
        color=RED,
        label="cos phi = 1.0 (11 MW -> +8.2 % violation)",
    )
    ax2.plot(
        d_km,
        f1_sp["regimes"]["cos_phi_fixed"][p_ref]["v_pu"],
        marker="s",
        ms=4,
        lw=1.3,
        color=WARM,
        label="Fixed cos phi = 0.95 ind (11 MW)",
    )
    ax2.plot(
        d_km,
        f1_sp["regimes"]["qu_droop"][p_ref]["v_pu"],
        marker="^",
        ms=4,
        lw=1.4,
        color=BLUE,
        label="NC RfG Q(U) droop (11 MW)",
    )
    ax2.plot(
        d_km,
        f1_sp["regimes"]["non_firm_70"]["16"]["v_pu"],
        marker="d",
        ms=4,
        lw=1.2,
        ls="--",
        color=GREEN,
        label="Q(U) + 70 % cap (16 MW installed)",
    )
    ax2.axhline(1.050, color="k", ls="--", lw=0.9, label="DSO PCC limit (+5.0 %, 1.05 pu)")
    ax2.axhline(1.020, color=GREY, ls=":", lw=0.8, label="Q(U) deadband upper edge (1.02 pu)")
    ax2.set_xlabel("Distance from GPZ 15 kV busbar [km]")
    ax2.set_ylabel("Node voltage U / Un [pu]")
    ax2.set_title("Feeder F1 (14.2 km AFL-6): voltage profile along line", fontsize=10)
    ax2.set_ylim(0.995, 1.112)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True)
    _save(fig, img_dir / "hosting_capacity.png")

    # 2. Sunny May week time-series on Feeder F1 (step(..., where='post', lw=0.9))
    fig, (ax_v, ax_q) = plt.subplots(2, 1, figsize=(10.0, 5.2), sharex=True)
    t_idx = pd.to_datetime(week_trace["timestamps"])
    m_unc = week_trace["modes"]["uncontrolled"]
    m_fix = week_trace["modes"]["cos_phi_fixed"]
    m_qu = week_trace["modes"]["qu_droop"]

    ax_v.step(t_idx, m_unc["v_pcc_pu"], where="post", lw=0.9, color=RED, label="Uncontrolled (cos phi = 1.0)")
    ax_v.step(t_idx, m_fix["v_pcc_pu"], where="post", lw=0.9, color=WARM, label="Fixed cos phi = 0.95 ind")
    ax_v.step(t_idx, m_qu["v_pcc_pu"], where="post", lw=1.0, color=BLUE, label="NC RfG Q(U) droop")
    ax_v.axhline(1.050, color="k", ls="--", lw=0.8, label="Max PCC voltage (1.050 pu)")
    ax_v.axhspan(0.980, 1.020, color=GREY, alpha=0.12, label="Q(U) deadband [0.98, 1.02] pu")
    ax_v.set_ylabel("PCC voltage [pu]")
    ax_v.set_ylim(0.982, 1.122)
    ax_v.set_title(
        f"Feeder F1 ({week_trace['p_inst_mw']:.1f} MW PV), 18-24 May 2026: PCC voltage and reactive power dispatch",
        fontsize=10,
    )
    ax_v.grid(alpha=0.25)
    ax_v.legend(loc="upper right", ncol=3, fontsize=7.8, frameon=True)

    ax_q.step(
        t_idx,
        m_fix["q_pcc_mvar"],
        where="post",
        lw=0.9,
        color=WARM,
        label="Q at PCC: Fixed cos phi = 0.95 [Mvar]",
    )
    ax_q.step(
        t_idx,
        m_qu["q_pcc_mvar"],
        where="post",
        lw=1.0,
        color=BLUE,
        label="Q at PCC: NC RfG Q(U) droop [Mvar]",
    )
    ax_q.step(
        t_idx,
        [x / 1000.0 for x in m_qu["loss_kw"]],
        where="post",
        lw=0.9,
        ls="--",
        color=INK,
        label="Feeder I^2 R active loss (Q(U)) [MW]",
    )
    ax_q.axhline(0.0, color="k", lw=0.5)
    ax_q.set_ylabel("Q [Mvar] / Loss [MW]")
    ax_q.set_ylim(-4.6, 2.2)
    ax_q.grid(alpha=0.25)
    ax_q.legend(loc="upper right", ncol=3, fontsize=7.8, frameon=True)
    _save(fig, img_dir / "voltage_qu_profile.png")

    # 3. Non-firm export cap trade-off (50 %..100 % of P_inst)
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    caps_pct = df_curt["export_cap_pct"].to_numpy()
    curt_pct = df_curt["curtailed_energy_pct"].to_numpy()
    bess_pct = df_curt["bess_2h_spilled_pct"].to_numpy()
    f1_mw = df_curt["f1_connectable_mw"].to_numpy()
    gpz_mw = df_curt["gpz_n1_connectable_mw"].to_numpy()

    ax_c1.plot(caps_pct, curt_pct, marker="o", ms=4.5, lw=1.4, color=ORANGE, label="Dynamic curtailment only")
    ax_c1.plot(
        caps_pct,
        bess_pct,
        marker="s",
        ms=4.5,
        lw=1.4,
        color=BLUE,
        label="With co-located 2 h BESS (peak-shaving)",
    )
    ax_c1.axvline(70, color=GREY, ls="--", lw=0.9, label="70 % export cap (2.4 % curtailed MWh)")
    ax_c1.set_xlabel("Active power export cap [% of P_inst]")
    ax_c1.set_ylabel("Annual spilled / curtailed PV energy [% of MWh]")
    ax_c1.set_title("Non-firm export cap vs annual energy loss", fontsize=10)
    ax_c1.set_xlim(102, 48)
    ax_c1.set_ylim(-0.4, 16.5)
    ax_c1.grid(alpha=0.25)
    ax_c1.legend(loc="upper left", fontsize=8, frameon=True)

    ax_c2.plot(caps_pct, gpz_mw, marker="^", ms=4.5, lw=1.4, color=NAVY, label="GPZ 110/15 kV (N-1: 25 MVA)")
    ax_c2.plot(caps_pct, f1_mw, marker="o", ms=4.5, lw=1.4, color=GREEN, label="Feeder F1 (with NC RfG Q(U))")
    ax_c2.axvline(70, color=GREY, ls="--", lw=0.9, label="70 % export cap (+43 % MW)")
    ax_c2.set_xlabel("Active power export cap [% of P_inst]")
    ax_c2.set_ylabel("Connectable renewable capacity [MW]")
    ax_c2.set_title("Unlocked connectable MW vs export cap", fontsize=10)
    ax_c2.set_xlim(102, 48)
    ax_c2.set_ylim(8, 52)
    ax_c2.grid(alpha=0.25)
    ax_c2.legend(loc="upper left", fontsize=8, frameon=True)
    _save(fig, img_dir / "curtailment_tradeoff.png")

    # 4. GPZ 110/15 kV transformer loading duration curve & seasonal heatmap under N-0 vs N-1
    fig, (ax_n1, ax_hm) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    hrs = n1_study["duration_hours"]
    ax_n1.plot(
        hrs,
        n1_study["s_mva_uncontrolled_21p5mw"],
        lw=1.3,
        color=GREY,
        label="21.5 MW PV (firm uncontrolled)",
    )
    ax_n1.plot(
        hrs,
        n1_study["s_mva_nonfirm_31mw"],
        lw=1.3,
        color=GREEN,
        label="31.0 MW PV (70 % export cap + Q(U))",
    )
    ax_n1.plot(
        hrs,
        n1_study["s_mva_pooling_bess_36p5mw"],
        lw=1.5,
        color=BLUE,
        label="36.5 MW PV+Wind + 2 h BESS pooling",
    )
    ax_n1.axhline(25.0, color=RED, ls="--", lw=1.0, label="N-1 limit: 1x 25 MVA transformer")
    ax_n1.set_xlabel("Hours per year [h]")
    ax_n1.set_ylabel("GPZ transformer apparent power S [MVA]")
    ax_n1.set_title("GPZ 110/15 kV loading duration curve (8,760 h)", fontsize=10)
    ax_n1.set_xlim(0, 8760)
    ax_n1.set_ylim(0, 36.5)
    ax_n1.grid(alpha=0.25)
    ax_n1.legend(loc="upper right", fontsize=7.8, frameon=True)

    hm_arr = np.asarray(n1_study["heatmap_p95_mva"], dtype=float)
    im = ax_hm.imshow(hm_arr, aspect="auto", cmap="YlGnBu", vmin=4.0, vmax=25.0, origin="upper")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("P95 S [MVA]", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax_hm.set_yticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    ax_hm.set_xticks([0, 4, 8, 12, 16, 20, 23], ["00", "04", "08", "12", "16", "20", "23"], fontsize=8)
    ax_hm.set_xlabel("Hour of day (Europe/Warsaw)")
    ax_hm.set_title("Seasonal P95 loading [MVA]: 36.5 MW pooling + BESS", fontsize=10)
    _save(fig, img_dir / "n1_transformer_loading.png")


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid connection studies and non-firm capacity on Polish MV/110 kV feeders</title>
<meta name="description" content="Steady-state and N-1 grid connection study of a Polish 110/15 kV substation (GPZ) and radial 15 kV feeders: voltage rise, thermal loading, NC RfG Type B/C Q(U) and cos-phi(P) reactive control, and how export caps or co-located storage unlock hosting capacity.">
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --accent: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --bad: #c8423b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --accent: #3987e5;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --bad: #d4574f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --accent: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --bad: #d4574f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 6px; }
h2 { font-size: 1.2rem; margin: 36px 0 6px; }
p { color: var(--ink2); margin: 6px 0 12px; max-width: 84ch; }
a { color: var(--accent); }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0; }
.tile .v { font-size: 1.45rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--ink2); font-size: 0.86rem; }
.plot { width: 100%; height: 420px; }
.controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; align-items: center;
            margin: 12px 0; color: var(--ink2); font-size: 0.9rem; }
select, input[type="range"] { font: inherit; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--ring);
                              background: var(--surface); color: var(--ink); width: 100%; }
.controls label { display: flex; flex-direction: column; gap: 4px; font-weight: 500; color: var(--ink); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
th, td { border-bottom: 1px solid var(--grid); padding: 6px 8px; text-align: left; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.note { font-size: 0.84rem; color: var(--muted); }
.warn { border-left: 3px solid var(--s2); padding-left: 10px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
.badge-ok { color: var(--s3); font-weight: 600; }
.badge-bad { color: var(--bad); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>Grid connection studies and non-firm capacity on Polish MV/110 kV feeders</h1>
<p>Steady-state and N-1 grid connection study of a Polish 110/15 kV substation (GPZ, two parallel 25 MVA OLTC
transformers) and four radial 15 kV feeders: how conductor R/X ratio, NC RfG Type B/C autonomous Q(U) and
cos-phi(P) reactive power control, non-firm active power export caps, and co-located BESS cable pooling govern
connectable renewable capacity. Code and method:
<a href="https://github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg">github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg</a>.</p>

<div class="tiles" id="kpi-tiles"></div>

<h2>1. Interactive feeder connection &amp; NC RfG simulator</h2>
<p>Select a 15 kV feeder, an NC RfG reactive / active power control regime, and candidate installed PV capacity
(2 MW to 20 MW). The left chart shows the exact Newton-Raphson AC voltage profile along the feeder against the
+5.0 % Polish DSO PCC rise ceiling (1.050 pu); the right chart shows the 7-day sunny May time-series scaled to
the selected capacity.</p>

<div class="card">
  <div class="controls">
    <label>15 kV Feeder
      <select id="sel-feeder">
        <option value="F1">F1 Rural Overhead 14.2 km (AFL-6 70/50, R/X = 0.95)</option>
        <option value="F2">F2 Mixed 9.6 km (NA2XS(F)2Y 150 + AFL-6 70, R/X = 1.10)</option>
        <option value="F3">F3 Suburban Cable 6.4 km (NA2XS(F)2Y 240, R/X = 0.38)</option>
      </select>
    </label>
    <label>Control Mode
      <select id="sel-regime">
        <option value="uncontrolled">1. cos phi = 1.0 (uncontrolled firm)</option>
        <option value="cos_phi_fixed">2. Fixed cos phi = 0.95 ind (above 50 % Pn)</option>
        <option value="qu_droop" selected>3. NC RfG Q(U) droop (deadband 0.98-1.02 pu)</option>
        <option value="non_firm_70">4. Q(U) + 70 % Non-Firm Export Cap</option>
        <option value="cable_pooling_bess">5. Q(U) + 2 h BESS Cable Pooling</option>
      </select>
    </label>
    <label>Candidate Installed PV Capacity: <span id="lbl-mw">11 MW</span>
      <input type="range" id="rng-mw" min="2" max="20" step="1" value="11">
    </label>
  </div>
  <div class="tiles" id="sim-kpis" style="margin: 10px 0 6px;"></div>
</div>

<div class="two" style="margin-top: 14px;">
  <div class="card"><div id="plot-spatial" class="plot"></div></div>
  <div class="card"><div id="plot-week" class="plot"></div></div>
</div>

<h2>2. Hosting capacity across Feeders F1-F3 and GPZ N-1</h2>
<p>Maximum connectable capacity (MW) satisfying both steady-state PCC voltage rise (U &le; 1.050 pu), continuous
conductor ampacity (I &le; I_dd), and 110/15 kV transformer N-1 backfeed (S &le; 25.0 MVA).</p>
<div class="card"><div id="plot-hc" class="plot"></div></div>
<div class="card scroll" style="margin-top: 12px;"><table id="tbl-hc"></table></div>

<h2>3. Non-firm export cap vs curtailed energy and GPZ N-1 transformer loading</h2>
<p>Capping peak active export to 70 % of installed MW sacrifices 2.4 % of annual PV generation while unlocking
+43 % more connectable capacity on both voltage-limited rural feeders and N-1 transformer-limited substations.
Co-located 2 h BESS cable pooling absorbs midday peaks and cuts spilled energy below 0.6 %.</p>
<div class="two">
  <div class="card"><div id="plot-curt" class="plot"></div></div>
  <div class="card"><div id="plot-n1" class="plot"></div></div>
</div>

<h2>4. Solver verification against exact analytical 2-bus solution</h2>
<p>The polar Newton-Raphson Ybus solver is checked against the closed-form biquadratic 2-bus AC power flow
solution and exact nodal power conservation.</p>
<div class="card scroll"><table id="tbl-ver"></table></div>

<p class="note" style="margin-top: 28px;">Standards &amp; grid code reference: Commission Regulation (EU) 2016/631
(NC RfG) Type B/C reactive power capability, PN-EN 50160 voltage quality, and Polish DSO (IRiESD / PTPiREE)
medium-voltage connection evaluation practice. Code: MIT licence.</p>
</main>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="data.js"></script>
<script>
const D = window.DATA;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 1) => x == null || isNaN(x) ? "-" : Number(x).toFixed(d);
const cfg = {responsive: true, displaylogo: false};

function layoutBase(extra) {
  return Object.assign({
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {family: "system-ui, -apple-system, Segoe UI, sans-serif", color: css("--ink2"), size: 12},
    margin: {l: 56, r: 18, t: 32, b: 48},
    hoverlabel: {bgcolor: css("--surface"), font: {color: css("--ink")}},
    legend: {orientation: "h", y: 1.12, x: 0, font: {size: 11}}
  }, extra);
}
const axis = extra => Object.assign({gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false}, extra);

function renderHeadlineTiles() {
  const H = D.headline;
  const items = [
    [`${fmt(H.f1_uncontrolled_mw, 1)} -> ${fmt(H.f1_qu_mw, 1)} MW`, "Feeder F1 (14.2 km) hosting capacity: uncontrolled vs NC RfG Q(U) (+84 %)"],
    [`-${fmt(H.reactive_loss_reduction_pct, 0)} %`, "Annual reactive I²R line losses on F1: Q(U) deadband vs fixed cos phi = 0.95"],
    [`${fmt(H.f1_nonfirm70_mw, 1)} MW (${fmt(H.curt_70_pct, 1)} % curt.)`, "Feeder F1 with 70 % non-firm export cap (+43 % MW for 2.4 % MWh)"],
    [`${fmt(H.gpz_n1_uncontrolled_mw, 1)} -> ${fmt(H.gpz_n1_pooling_bess_mw, 1)} MW`, "GPZ 110/15 kV N-1 (25 MVA) capacity with PV + Wind + 2 h BESS pooling"],
  ];
  document.getElementById("kpi-tiles").innerHTML = items.map(x =>
    `<div class="card tile"><div class="v">${x[0]}</div><div class="l">${x[1]}</div></div>`
  ).join("");
}

function updateInteractiveSim() {
  const fcode = document.getElementById("sel-feeder").value;
  const rcode = document.getElementById("sel-regime").value;
  const mw = document.getElementById("rng-mw").value;
  document.getElementById("lbl-mw").textContent = `${mw} MW`;

  const fdata = D.spatial[fcode];
  const pt = fdata.regimes[rcode][mw];
  const ptUnc = fdata.regimes["uncontrolled"][mw];
  const ptQu = fdata.regimes["qu_droop"][mw];

  const hcRow = D.hosting_table.find(r => r.feeder === fcode && r.regime === rcode);
  const vOk = pt.v_max_pu <= 1.050;
  const iOk = pt.max_loading_pct <= 100.0;
  const statusHtml = (vOk && iOk)
    ? `<span class="badge-ok">PASS (Feasible)</span>`
    : `<span class="badge-bad">VIOLATION (${!vOk ? "Voltage > 1.05 pu" : "Thermal > 100 %"})</span>`;

  const simTiles = [
    [statusHtml, `Connection verdict at ${mw} MW (Max connectable in mode: ${fmt(hcRow.hosting_capacity_mw, 1)} MW)`],
    [`${fmt(pt.v_max_pu, 4)} pu (${fmt((pt.v_max_pu - 1) * 100, 2)} %)`, "Peak PCC voltage U / Un (limit: 1.0500 pu / +5.0 %)"],
    [`${fmt(pt.max_loading_pct, 1)} %`, "Peak conductor thermal loading I / I_dd (limit: 100 %)"],
    [`${fmt(pt.q_mvar, 2)} Mvar / ${fmt(pt.loss_kw, 0)} kW`, "Net PCC reactive power Q and feeder active I²R loss"],
  ];
  document.getElementById("sim-kpis").innerHTML = simTiles.map(x =>
    `<div class="card tile"><div class="v">${x[0]}</div><div class="l">${x[1]}</div></div>`
  ).join("");

  // Plot spatial voltage profile along selected feeder
  const tracesSp = [
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: ptUnc.v_pu,
     name: "cos phi = 1.0 (uncontrolled)", line: {color: css("--muted"), dash: "dot", width: 2}},
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: ptQu.v_pu,
     name: "NC RfG Q(U) droop", line: {color: css("--s1"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: pt.v_pu,
     name: `Selected mode (${mw} MW)`, line: {color: css("--s2"), width: 3}, marker: {size: 8}},
  ];
  Plotly.react("plot-spatial", tracesSp, layoutBase({
    title: {text: `${fdata.name}: voltage profile at ${mw} MW`, font: {size: 13}},
    xaxis: axis({title: {text: "Distance from GPZ 15 kV busbar [km]"}}),
    yaxis: axis({title: {text: "Voltage U / Un [pu]"}, range: [0.99, 1.11]}),
    shapes: [
      {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.05, y1: 1.05, line: {color: css("--bad"), dash: "dash", width: 1.5}},
      {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.02, y1: 1.02, line: {color: css("--axis"), dash: "dot", width: 1}}
    ]
  }), cfg);

  // Plot sunny May week trace scaled by selected capacity relative to reference 11.4 MW
  const W = D.week_trace;
  const scale = Number(mw) / W.p_inst_mw;
  const vScale = arr => arr.map(v => Number((1.0 + (v - 1.0) * scale).toFixed(4)));
  const qScale = arr => arr.map(q => Number((q * scale).toFixed(2)));

  const tracesWk = [
    {type: "scatter", mode: "lines", x: W.timestamps, y: vScale(W.modes.uncontrolled.v_pcc_pu),
     name: "U_PCC: cos phi = 1.0", line: {color: css("--muted"), width: 1.3, shape: "hv"}},
    {type: "scatter", mode: "lines", x: W.timestamps, y: vScale(W.modes[rcode].v_pcc_pu),
     name: "U_PCC: Selected mode", line: {color: css("--s1"), width: 2.0, shape: "hv"}},
    {type: "scatter", mode: "lines", x: W.timestamps, y: qScale(W.modes[rcode].q_pcc_mvar),
     name: "Q_PCC [Mvar] (right)", yaxis: "y2", line: {color: css("--s2"), width: 1.5, dash: "dot", shape: "hv"}},
  ];
  Plotly.react("plot-week", tracesWk, layoutBase({
    title: {text: `Sunny May week (18-24 May 2026) at ${mw} MW`, font: {size: 13}},
    xaxis: axis({type: "date"}),
    yaxis: axis({title: {text: "PCC Voltage [pu]"}, range: [0.985, 1.11]}),
    yaxis2: axis({overlaying: "y", side: "right", showgrid: false, title: {text: "Q [Mvar]"}, range: [-6.5, 2.5]}),
    margin: {l: 56, r: 54, t: 32, b: 48},
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.05, y1: 1.05, line: {color: css("--bad"), dash: "dash", width: 1.2}}]
  }), cfg);
}

function renderStaticChartsAndTables() {
  const locs = ["F1", "F2", "F3", "GPZ_N1"];
  const locNames = ["F1 Rural (14.2 km)", "F2 Mixed (9.6 km)", "F3 Cable (6.4 km)", "GPZ 110/15 kV (N-1)"];
  const regOrder = [
    ["uncontrolled", "1. cos phi = 1.0", "#898781"],
    ["cos_phi_fixed", "2. Fixed cos phi = 0.95", "#e8a33d"],
    ["qu_droop", "3. NC RfG Q(U)", "#2a78d6"],
    ["non_firm_70", "4. Q(U) + 70 % cap", "#1baf7a"],
    ["cable_pooling_bess", "5. Q(U) + 2 h BESS", "#1b3f8f"],
  ];
  const hcBars = regOrder.map(([rc, rl, col]) => ({
    type: "bar", name: rl, x: locNames,
    y: locs.map(l => (D.hosting_table.find(r => r.feeder === l && r.regime === rc) || {}).hosting_capacity_mw || 0),
    marker: {color: col}
  }));
  Plotly.react("plot-hc", hcBars, layoutBase({
    barmode: "group",
    yaxis: axis({title: {text: "Connectable capacity [MW]"}, range: [0, 42]}),
  }), cfg);

  // Hosting capacity table
  const tHc = document.getElementById("tbl-hc");
  tHc.innerHTML = "<thead><tr><th>Location</th><th>Regime</th><th class='n'>Connectable [MW]</th><th class='n'>Max U_PCC [pu]</th><th class='n'>Max Loading [%]</th><th class='n'>Net Q [Mvar]</th><th class='n'>Loss [kW]</th><th>Binding Constraint</th></tr></thead><tbody>" +
    D.hosting_table.map(r => `<tr><td>${r.feeder}</td><td>${r.regime}</td><td class='n'>${fmt(r.hosting_capacity_mw, 2)}</td><td class='n'>${fmt(r.v_pcc_pu, 4)}</td><td class='n'>${fmt(r.max_branch_loading_pct, 1)}</td><td class='n'>${fmt(r.q_net_mvar, 2)}</td><td class='n'>${fmt(r.feeder_loss_kw, 1)}</td><td>${r.binding_constraint}</td></tr>`).join("") +
    "</tbody>";

  // Curtailment trade-off chart
  const C = D.curtailment;
  Plotly.react("plot-curt", [
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.curtailed_energy_pct),
     name: "Curtailed PV energy [%]", line: {color: css("--s2"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.bess_2h_spilled_pct),
     name: "Spilled with 2 h BESS [%]", line: {color: css("--s3"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.gpz_n1_connectable_mw),
     name: "GPZ N-1 Connectable [MW] (right)", yaxis: "y2", line: {color: css("--s1"), width: 2, dash: "dash"}},
  ], layoutBase({
    title: {text: "Export cap vs curtailed energy & connectable MW", font: {size: 13}},
    xaxis: axis({title: {text: "Export cap [% of P_inst]"}, autorange: "reversed"}),
    yaxis: axis({title: {text: "Curtailed energy [% of MWh]"}, range: [0, 16]}),
    yaxis2: axis({overlaying: "y", side: "right", showgrid: false, title: {text: "GPZ N-1 MW"}, range: [15, 48]}),
    margin: {l: 56, r: 54, t: 32, b: 48}
  }), cfg);

  // N-1 transformer loading duration curve
  const N1 = D.n1_study;
  Plotly.react("plot-n1", [
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_uncontrolled_21p5mw,
     name: "21.5 MW PV (firm)", line: {color: css("--muted"), width: 2}},
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_nonfirm_31mw,
     name: "31.0 MW PV (70 % cap)", line: {color: css("--s3"), width: 2}},
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_pooling_bess_36p5mw,
     name: "36.5 MW PV+Wind + BESS", line: {color: css("--s1"), width: 2.2}},
  ], layoutBase({
    title: {text: "GPZ 110/15 kV transformer S [MVA] duration curve", font: {size: 13}},
    xaxis: axis({title: {text: "Hours per year [h]"}}),
    yaxis: axis({title: {text: "Apparent power S [MVA]"}, range: [0, 30]}),
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 25.0, y1: 25.0, line: {color: css("--bad"), dash: "dash", width: 1.5}}]
  }), cfg);

  // Verification table
  const V = D.verification;
  const vRows = [
    ["2-bus PCC voltage magnitude |V1| [pu]", V.v1_exact_pu.toFixed(10), V.v1_nr_pu.toFixed(10), V.v_abs_error_pu.toExponential(2)],
    ["2-bus PCC voltage angle theta1 [rad]", V.theta1_exact_rad.toFixed(10), V.theta1_nr_rad.toFixed(10), V.theta_abs_error_rad.toExponential(2)],
    ["Active power balance (P_slack + P_gen - P_load - P_loss) [MW]", "0.0000000000", V.p_balance_error_mw.toExponential(2), V.p_balance_error_mw.toExponential(2)],
    ["Reactive power balance (Q_slack + Q_gen - Q_load - Q_loss) [Mvar]", "0.0000000000", V.q_balance_error_mvar.toExponential(2), V.q_balance_error_mvar.toExponential(2)],
  ];
  document.getElementById("tbl-ver").innerHTML =
    "<thead><tr><th>Verification Quantity</th><th class='n'>Analytical Exact</th><th class='n'>Newton-Raphson Ybus</th><th class='n'>Absolute Error</th></tr></thead><tbody>" +
    vRows.map(r => `<tr><td>${r[0]}</td><td class='n'>${r[1]}</td><td class='n'>${r[2]}</td><td class='n'>${r[3]}</td></tr>`).join("") +
    "</tbody>";
}

function renderAll() {
  renderHeadlineTiles();
  updateInteractiveSim();
  renderStaticChartsAndTables();
}

document.getElementById("sel-feeder").addEventListener("change", updateInteractiveSim);
document.getElementById("sel-regime").addEventListener("change", updateInteractiveSim);
document.getElementById("rng-mw").addEventListener("input", updateInteractiveSim);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 200); });
renderAll();
</script>
</body>
</html>
"""


def generate_dashboard_html(summary_payload: dict, docs_dir: Path) -> None:
    """Write docs/data.js, docs/index.html, and docs/.nojekyll."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    (docs_dir / "data.js").write_text(
        "window.DATA = " + json.dumps(summary_payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (docs_dir / "index.html").write_text(DASHBOARD_HTML, encoding="utf-8")


def run_full_pipeline(output_root: Path | None = None) -> dict:
    """Execute the full Polish MV/110 kV grid connection study and write results, plots, and dashboard."""
    root = output_root or REPO_ROOT
    results_dir = root / "results"
    docs_dir = root / "docs"
    img_dir = docs_dir / "img"
    results_dir.mkdir(parents=True, exist_ok=True)

    net_n1 = build_polish_gpz_network(n_transformers=1)
    net_n0 = build_polish_gpz_network(n_transformers=2)
    df_profiles = generate_annual_polish_profiles()

    df_hc = compute_hosting_capacity_table(net_n1)
    df_hc.to_csv(results_dir / "hosting_capacity.csv", index=False)

    f1_qu_mw = float(df_hc[(df_hc["feeder"] == "F1") & (df_hc["regime"] == "qu_droop")]["hosting_capacity_mw"].iloc[0])
    gpz_qu_mw = float(
        df_hc[(df_hc["feeder"] == "GPZ_N1") & (df_hc["regime"] == "qu_droop")]["hosting_capacity_mw"].iloc[0]
    )

    df_curt = compute_curtailment_tradeoff(df_profiles, f1_firm_qu_mw=f1_qu_mw, gpz_firm_qu_mw=gpz_qu_mw)
    df_curt.to_csv(results_dir / "curtailment_tradeoff.csv", index=False)

    loss_cmp = compute_annual_reactive_loss_comparison(net_n0, df_profiles, feeder_code="F1", p_inst_mw=6.2)
    n1_study = compute_n1_transformer_study(df_profiles)
    week_trace = simulate_sunny_week_trace(net_n0, df_profiles, feeder_code="F1", p_inst_mw=11.4)
    spatial_profiles = compute_spatial_voltage_profiles(net_n1)
    ver = verify_two_bus_analytical()

    (results_dir / "verification.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")

    def _hc_val(f: str, r: str) -> float:
        return float(df_hc[(df_hc["feeder"] == f) & (df_hc["regime"] == r)]["hosting_capacity_mw"].iloc[0])

    curt_70_row = df_curt[df_curt["export_cap_pct"] == 70].iloc[0]
    headline = {
        "f1_uncontrolled_mw": _hc_val("F1", "uncontrolled"),
        "f1_cosphi_mw": _hc_val("F1", "cos_phi_fixed"),
        "f1_qu_mw": _hc_val("F1", "qu_droop"),
        "f1_nonfirm70_mw": _hc_val("F1", "non_firm_70"),
        "f1_pooling_bess_mw": _hc_val("F1", "cable_pooling_bess"),
        "f2_uncontrolled_mw": _hc_val("F2", "uncontrolled"),
        "f2_cosphi_mw": _hc_val("F2", "cos_phi_fixed"),
        "f2_qu_mw": _hc_val("F2", "qu_droop"),
        "f2_nonfirm70_mw": _hc_val("F2", "non_firm_70"),
        "f3_uncontrolled_mw": _hc_val("F3", "uncontrolled"),
        "f3_cosphi_mw": _hc_val("F3", "cos_phi_fixed"),
        "f3_qu_mw": _hc_val("F3", "qu_droop"),
        "f3_nonfirm70_mw": _hc_val("F3", "non_firm_70"),
        "gpz_n1_uncontrolled_mw": _hc_val("GPZ_N1", "uncontrolled"),
        "gpz_n1_cosphi_mw": _hc_val("GPZ_N1", "cos_phi_fixed"),
        "gpz_n1_qu_mw": _hc_val("GPZ_N1", "qu_droop"),
        "gpz_n1_nonfirm70_mw": _hc_val("GPZ_N1", "non_firm_70"),
        "gpz_n1_pooling_bess_mw": _hc_val("GPZ_N1", "cable_pooling_bess"),
        "reactive_loss_reduction_pct": loss_cmp["reactive_loss_reduction_pct"],
        "curt_70_pct": float(curt_70_row["curtailed_energy_pct"]),
        "pooling_spilled_pct": n1_study["pooling_spilled_energy_pct"],
    }

    summary_payload = {
        "headline": headline,
        "hosting_table": df_hc.to_dict(orient="records"),
        "curtailment": df_curt.to_dict(orient="records"),
        "reactive_loss_comparison": loss_cmp,
        "n1_study": n1_study,
        "week_trace": week_trace,
        "spatial": spatial_profiles,
        "verification": ver,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    generate_plots(df_hc, spatial_profiles, week_trace, df_curt, n1_study, img_dir)
    generate_dashboard_html(summary_payload, docs_dir)
    return summary_payload
