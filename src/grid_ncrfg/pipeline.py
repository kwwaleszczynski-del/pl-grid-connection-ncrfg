"""End-to-end simulation pipeline, Matplotlib figure generator, and interactive Plotly dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .network import LINE_TYPES, Z_BASE_MV_OHM, NetworkModel, build_polish_gpz_network
from .powerflow import verify_two_bus_analytical
from .ncrfg import CONTROL_REGIMES, solve_feeder_with_ncrfg
from .profiles import PV_YEAR, generate_annual_polish_profiles, pv_dc_year
from .hosting import (
    GPZ_EXISTING_GEN_MW,
    compute_hosting_capacity_table,
    compute_annual_reactive_loss_comparison,
    compute_curtailment_tradeoff,
    compute_export_cap_by_pv_year,
    compute_n1_transformer_study,
    simulate_sunny_week_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HIGHLIGHT_CAP_PCT = 70
SPATIAL_CAPACITIES_MW = [round(0.5 * k, 1) for k in range(1, 33)]  # 0.5 .. 16.0 MW


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def compute_spatial_voltage_profiles(net: NetworkModel) -> dict:
    """Compute distance-voltage profiles U(x) and branch thermal loadings across F1, F2, F3 for the slider."""
    out: dict[str, dict] = {}
    for fcode in ["F1", "F2", "F3"]:
        fspec = net.feeders[fcode]
        bus_ids = [1] + list(fspec.bus_ids)
        br_ids = list(fspec.branch_ids)
        plant_buses = [fspec.pcc_bus_id, fspec.bus_ids[-2]]
        distances_km = [round(float(net.buses[b].distance_km), 2) for b in bus_ids]
        bus_names = [net.buses[b].name for b in bus_ids]

        regimes_data: dict[str, dict[str, dict]] = {}
        for rcode in CONTROL_REGIMES:
            regimes_data[rcode] = {}
            for p_mw in SPATIAL_CAPACITIES_MW:
                res = solve_feeder_with_ncrfg(net, fcode, p_mw, rcode)
                q_plant = sum(float(res.q_inj_mvar[b]) + net.buses[b].q_load_min_mvar for b in plant_buses)
                regimes_data[rcode][f"{p_mw:.1f}"] = {
                    "v_pu": [round(float(res.v_mag_pu[b]), 4) for b in bus_ids],
                    "v_max_pu": round(float(np.max(res.v_mag_pu[list(fspec.bus_ids)])), 4),
                    "max_loading_pct": round(float(np.max(res.branch_loading_pct[br_ids])), 1),
                    "loss_kw": round(float(np.sum(res.branch_p_loss_mw[br_ids]) * 1000.0), 1),
                    "q_mvar": round(q_plant, 2),
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


def describe_network(net: NetworkModel) -> dict:
    """Line-type data and feeder summaries actually used by the model (for the README and dashboard)."""
    line_types = [
        {
            "code": lt.code,
            "r_ohm_per_km": round(lt.r_ohm_per_km, 4),
            "x_ohm_per_km": round(lt.x_ohm_per_km, 4),
            "b_us_per_km": round(lt.b_us_per_km, 2),
            "i_dd_a": round(lt.i_dd_a, 0),
            "s_thermal_mva": round(lt.s_thermal_mva, 2),
            "rx_ratio": round(lt.rx_ratio, 2),
        }
        for lt in LINE_TYPES.values()
    ]
    feeders = []
    for fspec in net.feeders.values():
        br = [net.branches[k] for k in fspec.branch_ids]
        feeders.append(
            {
                "code": fspec.code,
                "description": fspec.description,
                "length_km": fspec.total_length_km,
                "r_ohm": round(sum(b.r_pu for b in br) * Z_BASE_MV_OHM, 3),
                "x_ohm": round(sum(b.x_pu for b in br) * Z_BASE_MV_OHM, 3),
                "rx_ratio": fspec.rx_ratio,
                "min_load_mw": fspec.p_load_min_mw,
                "peak_load_mw": fspec.p_load_peak_mw,
                "weakest_branch_mva": round(min(b.s_max_mva for b in br), 2),
            }
        )
    return {"line_types": line_types, "feeders": feeders}


def generate_plots(
    df_hc: pd.DataFrame,
    spatial_profiles: dict,
    spatial_ref_mw: str,
    week_trace: dict,
    df_curt: pd.DataFrame,
    n1_study: dict,
    img_dir: Path,
) -> None:
    """Generate the 4 README figures in docs/img/*.png."""
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
        "F1 overhead\n(14.2 km)",
        "F2 mixed\n(9.6 km)",
        "F3 cable\n(6.4 km)",
        "GPZ busbar\n(N-1: 25 MVA)",
    ]
    regime_order = [
        ("uncontrolled", "1. cos phi = 1.0", GREY),
        ("cos_phi_p", "2. cos phi(P), 0.95 ind at Pn", WARM),
        ("qu_droop", "3. Q(U) droop", BLUE),
    ]
    x = np.arange(len(locs))
    w = 0.26
    for i, (rcode, rlabel, col) in enumerate(regime_order):
        vals = [
            float(df_hc[(df_hc["feeder"] == loc) & (df_hc["regime"] == rcode)]["hosting_capacity_mw"].iloc[0])
            for loc in locs
        ]
        bars = ax1.bar(x + (i - 1) * w, vals, width=w, label=rlabel, color=col)
        ax1.bar_label(bars, fmt="%.1f", fontsize=8, padding=1.5)
    ax1.set_xticks(x, loc_labels, fontsize=9)
    ax1.set_ylabel("Connectable capacity [MW]")
    ax1.set_title("Hosting capacity by location and reactive control", fontsize=10)
    ax1.set_ylim(0, 1.15 * float(df_hc["hosting_capacity_mw"].max()))
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)

    f1_sp = spatial_profiles["F1"]
    d_km = f1_sp["distances_km"]
    styles = {"uncontrolled": ("o", RED), "cos_phi_p": ("s", WARM), "qu_droop": ("^", BLUE)}
    for rcode, (mk, col) in styles.items():
        pt = f1_sp["regimes"][rcode][spatial_ref_mw]
        ax2.plot(d_km, pt["v_pu"], marker=mk, ms=4, lw=1.3, color=col,
                 label=f"{CONTROL_REGIMES[rcode].short_label}: max {pt['v_max_pu']:.4f} pu")
    ax2.axhline(1.050, color="k", ls="--", lw=0.9, label="Voltage limit 1.05 pu")
    ax2.axhline(1.020, color=GREY, ls=":", lw=0.8, label="Q(U) deadband upper edge 1.02 pu")
    ax2.set_xlabel("Distance from GPZ 15 kV busbar [km]")
    ax2.set_ylabel("Node voltage U / Un [pu]")
    ax2.set_title(f"Feeder F1 at {float(spatial_ref_mw):.1f} MW: voltage along the line", fontsize=10)
    ax2.grid(alpha=0.25)
    # headroom above the 1.05 pu line for the legend; the profiles rise to the right, under it
    ax2.set_ylim(top=max(1.10, ax2.get_ylim()[1] + 0.03))
    ax2.legend(loc="upper left", fontsize=9, frameon=True)
    _save(fig, img_dir / "hosting_capacity.png")

    # 2. Sunniest May-June week time-series on Feeder F1
    fig, (ax_v, ax_q) = plt.subplots(2, 1, figsize=(10.0, 5.2), sharex=True)
    t_idx = pd.to_datetime(week_trace["timestamps"])
    m_unc = week_trace["modes"]["uncontrolled"]
    m_cos = week_trace["modes"]["cos_phi_p"]
    m_qu = week_trace["modes"]["qu_droop"]

    ax_v.step(t_idx, m_unc["v_pcc_pu"], where="post", lw=0.9, color=RED, label="cos phi = 1.0")
    ax_v.step(t_idx, m_cos["v_pcc_pu"], where="post", lw=0.9, color=WARM, label="cos phi(P)")
    ax_v.step(t_idx, m_qu["v_pcc_pu"], where="post", lw=1.0, color=BLUE, label="Q(U) droop")
    ax_v.axhline(1.050, color="k", ls="--", lw=0.8, label="Voltage limit 1.05 pu")
    ax_v.axhspan(0.980, 1.020, color=GREY, alpha=0.12, label="Q(U) deadband [0.98, 1.02] pu")
    v_all = np.concatenate([m_unc["v_pcc_pu"], m_cos["v_pcc_pu"], m_qu["v_pcc_pu"]])
    ax_v.set_ylim(min(0.975, float(v_all.min()) - 0.005), max(1.055, float(v_all.max())) + 0.03)
    ax_v.set_ylabel("PCC voltage [pu]")
    ax_v.set_title(
        f"Feeder F1 ({week_trace['p_inst_mw']:.2f} MW PV), {week_trace['label']}: PCC voltage and plant reactive power",
        fontsize=10,
    )
    ax_v.grid(alpha=0.25)
    ax_v.legend(loc="upper right", ncol=3, fontsize=9, frameon=True)

    ax_q.step(t_idx, m_cos["q_pcc_mvar"], where="post", lw=0.9, color=WARM, label="Q plant: cos phi(P) [Mvar]")
    ax_q.step(t_idx, m_qu["q_pcc_mvar"], where="post", lw=1.0, color=BLUE, label="Q plant: Q(U) [Mvar]")
    ax_q.step(t_idx, [x / 1000.0 for x in m_qu["loss_kw"]], where="post", lw=0.9, ls="--", color=INK,
              label="Feeder I^2 R loss, Q(U) [MW]")
    ax_q.axhline(0.0, color="k", lw=0.5)
    q_min = float(min(np.min(m_cos["q_pcc_mvar"]), np.min(m_qu["q_pcc_mvar"])))
    q_max = float(max(np.max(m_qu["loss_kw"]) / 1000.0, np.max(m_qu["q_pcc_mvar"]), 0.0))
    ax_q.set_ylim(q_min - 0.45 * (q_max - q_min), q_max + 0.1 * (q_max - q_min))
    ax_q.set_ylabel("Q [Mvar] / Loss [MW]")
    ax_q.grid(alpha=0.25)
    ax_q.legend(loc="lower right", ncol=3, fontsize=9, frameon=True)
    _save(fig, img_dir / "voltage_qu_profile.png")

    # 3. Export cap trade-off (50 %..100 % of P_inst)
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    caps_pct = df_curt["export_cap_pct"].to_numpy()
    row_hl = df_curt[df_curt["export_cap_pct"] == HIGHLIGHT_CAP_PCT].iloc[0]

    ax_c1.plot(caps_pct, df_curt["curtailed_energy_pct"], marker="o", ms=4.5, lw=1.4, color=ORANGE,
               label="Curtailment only")
    ax_c1.plot(caps_pct, df_curt["bess_lost_energy_pct"], marker="s", ms=4.5, lw=1.4, color=BLUE,
               label="With battery (0.35 P_inst, 2 h)")
    ax_c1.axvline(HIGHLIGHT_CAP_PCT, color=GREY, ls="--", lw=0.9,
                  label=f"{HIGHLIGHT_CAP_PCT} % cap: {row_hl['curtailed_energy_pct']:.2f} % of MWh")
    ax_c1.set_xlabel("Active power export cap [% of P_inst]")
    ax_c1.set_ylabel("Annual PV energy not delivered [% of MWh]")
    ax_c1.set_title("Export cap vs annual energy loss", fontsize=10)
    ax_c1.set_xlim(102, 48)
    ax_c1.grid(alpha=0.25)
    ax_c1.legend(loc="upper left", fontsize=9, frameon=True)

    ax_c2.plot(caps_pct, df_curt["energy_gain_vs_firm_pct"], marker="o", ms=4.5, lw=1.4, color=GREEN,
               label="Oversized PV, capped")
    ax_c2.plot(caps_pct, df_curt["bess_energy_gain_vs_firm_pct"], marker="s", ms=4.5, lw=1.4, color=NAVY,
               label="Oversized PV + battery")
    ax_c2.plot(caps_pct, 100.0 * (df_curt["installed_per_export_mw"] - 1.0), ls=":", lw=1.1, color=GREY,
               label="Installed MW gain, 1/cap - 1 (arithmetic)")
    ax_c2.axvline(HIGHLIGHT_CAP_PCT, color=GREY, ls="--", lw=0.9,
                  label=f"{HIGHLIGHT_CAP_PCT} % cap: +{row_hl['energy_gain_vs_firm_pct']:.1f} % MWh")
    ax_c2.set_xlabel("Active power export cap [% of P_inst]")
    ax_c2.set_ylabel("Gain vs firm plant on same export limit [%]")
    ax_c2.set_title("Annual energy through a fixed export limit", fontsize=10)
    ax_c2.set_xlim(102, 48)
    ax_c2.grid(alpha=0.25)
    ax_c2.legend(loc="upper left", fontsize=9, frameon=True)
    _save(fig, img_dir / "curtailment_tradeoff.png")

    # 4. GPZ 110/15 kV transformer loading duration curve & seasonal heatmap under N-1
    fig, (ax_n1, ax_hm) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    hrs = n1_study["duration_hours"]
    s_lim = n1_study["s_limit_mva"]
    ax_n1.plot(hrs, n1_study["s_mva_firm_pv"], lw=1.3, color=GREY,
               label=f"A: {n1_study['p_firm_mw']:.2f} MW PV, firm")
    ax_n1.plot(hrs, n1_study["s_mva_capped_pv"], lw=1.3, color=GREEN,
               label=f"B: {n1_study['capped_pv_mw']:.2f} MW PV, export cap {n1_study['p_firm_mw']:.2f} MW")
    ax_n1.plot(hrs, n1_study["s_mva_pooling_bess"], lw=1.5, color=BLUE,
               label=f"C: + {n1_study['pooling_wind_mw']:.2f} MW wind + "
                     f"{n1_study['pooling_bess_mw']:.2f} MW / {n1_study['pooling_bess_mwh']:.2f} MWh BESS")
    ax_n1.axhline(s_lim, color=RED, ls="--", lw=1.0, label=f"N-1 limit: 1 x {s_lim:.0f} MVA")
    ax_n1.set_xlabel("Hours per year [h]")
    ax_n1.set_ylabel("GPZ transformer apparent power S [MVA]")
    ax_n1.set_title("GPZ 110/15 kV N-1 loading duration curve (8,760 h)", fontsize=10)
    ax_n1.set_xlim(0, hrs[-1] + 1)
    ax_n1.set_ylim(0, 1.25 * s_lim)
    ax_n1.grid(alpha=0.25)
    # just below the N-1 limit line, right of the knee where all curves are low
    ax_n1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.77), fontsize=9, frameon=True)

    hm_arr = np.asarray(n1_study["heatmap_p95_mva"], dtype=float)
    im = ax_hm.imshow(hm_arr, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=s_lim, origin="upper")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("P95 S [MVA]", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    ax_hm.set_yticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=9)
    ax_hm.set_xticks([0, 4, 8, 12, 16, 20, 23], ["00", "04", "08", "12", "16", "20", "23"], fontsize=9)
    ax_hm.set_xlabel("Hour of day (Europe/Warsaw)")
    ax_hm.set_title("Scenario C: monthly P95 transformer loading [MVA]", fontsize=10)
    _save(fig, img_dir / "n1_transformer_loading.png")


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid connection studies and non-firm capacity on Polish MV/110 kV feeders</title>
<meta name="description" content="Steady-state and N-1 grid connection study of a synthetic Polish 110/15 kV substation (GPZ) and radial 15 kV feeders: voltage rise, thermal loading, NC RfG cos-phi(P) and Q(U) reactive control, export caps and co-located storage.">
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
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(380px, 100%), 1fr)); gap: 14px; }
.badge-ok { color: var(--s3); font-weight: 600; }
.badge-bad { color: var(--bad); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>Grid connection studies and non-firm capacity on Polish MV/110 kV feeders</h1>
<p>Steady-state and N-1 study of a synthetic Polish 110/15 kV substation (GPZ, two 25 MVA OLTC transformers) and
radial 15 kV feeders built from catalogue AFL-6 and NA2XS(F)2Y data: how feeder impedance, NC RfG cos-phi(P) and
Q(U) reactive control, active power export caps and a co-located battery change connectable capacity and annual
energy. Every number on this page is read from the pipeline output. Code and method:
<a href="https://github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg">github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg</a>.</p>

<div class="tiles" id="kpi-tiles"></div>

<h2>1. Feeder connection simulator</h2>
<p>Select a 15 kV feeder, a reactive control regime and a candidate PV capacity. The left chart is the Newton-Raphson
AC voltage profile along the feeder at minimum load against the 1.05 pu limit (GPZ busbar held at 1.00 pu by the
OLTC). The right chart is the hourly power flow of the sunniest May-June week on Feeder F1 for the plant size stated in its
title (it does not change with the slider).</p>

<div class="card">
  <div class="controls">
    <label>15 kV feeder
      <select id="sel-feeder"></select>
    </label>
    <label>Control regime
      <select id="sel-regime">
        <option value="uncontrolled">1. cos phi = 1.0</option>
        <option value="cos_phi_p">2. cos phi(P): 0.95 ind at Pn</option>
        <option value="qu_droop" selected>3. Q(U) droop (deadband 0.98-1.02 pu)</option>
      </select>
    </label>
    <label>Candidate PV capacity: <span id="lbl-mw"></span>
      <input type="range" id="rng-mw" min="0" step="1" value="5">
    </label>
  </div>
  <div class="tiles" id="sim-kpis" style="margin: 10px 0 6px;"></div>
</div>

<div class="two" style="margin-top: 14px;">
  <div class="card"><div id="plot-spatial" class="plot"></div></div>
  <div class="card"><div id="plot-week" class="plot"></div></div>
</div>

<h2>2. Hosting capacity across Feeders F1-F3 and the GPZ busbar under N-1</h2>
<p>Largest plant (MW) for which bus voltages stay at or below 1.05 pu and every branch within its continuous
ampacity, at minimum load and full plant output. The GPZ row is new generation at the 15 kV busbar with one
25 MVA transformer out and existing busbar PV at full output; there the binding limit is the transformer rating.</p>
<div class="card"><div id="plot-hc" class="plot"></div></div>
<div class="card scroll" style="margin-top: 12px;"><table id="tbl-hc"></table></div>

<h2>3. Export cap energy trade-off and GPZ N-1 loading over a year</h2>
<p id="txt-curt"></p>
<div class="two">
  <div class="card"><div id="plot-curt" class="plot"></div></div>
  <div class="card"><div id="plot-n1" class="plot"></div></div>
</div>

<h2>4. Solver verification against the closed-form 2-bus solution</h2>
<p>The polar Newton-Raphson Ybus solver is checked against the closed-form biquadratic 2-bus AC power flow
solution and nodal power conservation.</p>
<div class="card scroll"><table id="tbl-ver"></table></div>

<p class="note" style="margin-top: 28px;">Synthetic network, load and wind; PV from PVGIS. Not a real substation or a DSO assessment.
Control modes follow NC RfG (Commission Regulation (EU) 2016/631) and EN 50549-2 families; the parameter values and
the 1.05 pu limit are study assumptions to be verified against the current Polish requirements of general
application and the DSO IRiESD. Code: MIT licence.</p>
</main>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="data.js"></script>
<script>
const D = window.DATA;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 1) => x == null || isNaN(x) ? "-" : Number(x).toFixed(d);
const cfg = {responsive: true, displaylogo: false};
const CAPS = D.spatial_capacities_mw;
const REG = {uncontrolled: "cos phi = 1.0", cos_phi_p: "cos phi(P)", qu_droop: "Q(U) droop"};

function layoutBase(extra) {
  const out = Object.assign({
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {family: "system-ui, -apple-system, Segoe UI, sans-serif", color: css("--ink2"), size: 12},
    margin: {l: 56, r: 18, t: 32, b: 48},
    hoverlabel: {bgcolor: css("--surface"), font: {color: css("--ink")}},
    legend: {orientation: "h", y: 1.14, x: 0, font: {size: 11}}
  }, extra);
  if (out.title) {
    // title at the top of the figure, the legend in its own band between title and plot (wraps on a phone)
    const narrow = window.innerWidth < 640;
    out.title = Object.assign({y: 1, yref: "container", yanchor: "top", pad: {t: 6}}, out.title);
    out.legend = Object.assign({}, out.legend, {y: 1.01, yanchor: "bottom"});
    out.margin = Object.assign({}, out.margin, {t: narrow ? 104 : 64});
  }
  return out;
}
const axis = extra => Object.assign({gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false}, extra);

function renderHeadlineTiles() {
  const H = D.headline;
  const items = [
    [`${fmt(H.f1_uncontrolled_mw, 2)} -> ${fmt(H.f1_qu_mw, 2)} MW`,
     `Feeder F1 (14.2 km overhead) hosting capacity, cos phi = 1 vs Q(U) (${H.f1_qu_gain_pct >= 0 ? "+" : ""}${fmt(H.f1_qu_gain_pct, 0)} %)`],
    [`-${fmt(H.q_reduction_pct, 0)} %`,
     `Annual reactive energy absorbed on F1, Q(U) vs cos phi(P), ${fmt(H.reactive_study_p_mw, 2)} MW plant`],
    [`${fmt(H.curt_hl_pct, 2)} % of MWh`,
     `Annual PV energy curtailed by a ${H.highlight_cap_pct} % export cap (+${fmt(H.energy_gain_hl_pct, 1)} % MWh through the same export limit)`],
    [`${fmt(H.gpz_n1_uncontrolled_mw, 2)} MW`,
     `GPZ busbar hosting capacity under N-1 (one 25 MVA transformer), cos phi = 1`],
  ];
  document.getElementById("kpi-tiles").innerHTML = items.map(x =>
    `<div class="card tile"><div class="v">${x[0]}</div><div class="l">${x[1]}</div></div>`
  ).join("");
}

function initControls() {
  const sel = document.getElementById("sel-feeder");
  sel.innerHTML = ["F1", "F2", "F3"].map(f =>
    `<option value="${f}">${D.spatial[f].description} (R/X = ${fmt(D.spatial[f].rx_ratio, 2)})</option>`).join("");
  const rng = document.getElementById("rng-mw");
  rng.max = CAPS.length - 1;
  rng.value = Math.max(0, CAPS.findIndex(c => c >= D.headline.f1_uncontrolled_mw));
}

function updateInteractiveSim() {
  const fcode = document.getElementById("sel-feeder").value;
  const rcode = document.getElementById("sel-regime").value;
  const mw = CAPS[Number(document.getElementById("rng-mw").value)];
  const key = mw.toFixed(1);
  document.getElementById("lbl-mw").textContent = `${key} MW`;

  const fdata = D.spatial[fcode];
  const pt = fdata.regimes[rcode][key];
  const ptUnc = fdata.regimes["uncontrolled"][key];
  const ptQu = fdata.regimes["qu_droop"][key];

  const hcRow = D.hosting_table.find(r => r.feeder === fcode && r.regime === rcode);
  const vOk = pt.v_max_pu <= 1.050;
  const iOk = pt.max_loading_pct <= 100.0;
  const statusHtml = (vOk && iOk)
    ? `<span class="badge-ok">Within limits</span>`
    : `<span class="badge-bad">Violation (${!vOk ? "voltage > 1.05 pu" : "thermal > 100 %"})</span>`;

  const simTiles = [
    [statusHtml, `At ${key} MW (hosting capacity in this regime: ${fmt(hcRow.hosting_capacity_mw, 2)} MW)`],
    [`${fmt(pt.v_max_pu, 4)} pu`, "Highest feeder voltage (limit 1.0500 pu)"],
    [`${fmt(pt.max_loading_pct, 1)} %`, "Highest branch loading I / I_dd (limit 100 %)"],
    [`${fmt(pt.q_mvar, 2)} Mvar / ${fmt(pt.loss_kw, 0)} kW`, "Plant reactive power and feeder I²R loss"],
  ];
  document.getElementById("sim-kpis").innerHTML = simTiles.map(x =>
    `<div class="card tile"><div class="v">${x[0]}</div><div class="l">${x[1]}</div></div>`
  ).join("");

  const tracesSp = [
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: ptUnc.v_pu,
     name: "cos phi = 1.0", line: {color: css("--muted"), dash: "dot", width: 2}},
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: ptQu.v_pu,
     name: "Q(U) droop", line: {color: css("--s1"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: fdata.distances_km, y: pt.v_pu,
     name: `Selected: ${REG[rcode]}`, line: {color: css("--s2"), width: 3}, marker: {size: 8}},
  ];
  Plotly.react("plot-spatial", tracesSp, layoutBase({
    title: {text: `${fcode}: voltage profile at ${key} MW`, font: {size: 13}},
    xaxis: axis({title: {text: "Distance from GPZ 15 kV busbar [km]"}}),
    yaxis: axis({title: {text: "Voltage U / Un [pu]"}}),
    shapes: [
      {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.05, y1: 1.05, line: {color: css("--bad"), dash: "dash", width: 1.5}},
      {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.02, y1: 1.02, line: {color: css("--axis"), dash: "dot", width: 1}}
    ]
  }), cfg);

  const W = D.week_trace;
  const tracesWk = [
    {type: "scatter", mode: "lines", x: W.timestamps, y: W.modes.uncontrolled.v_pcc_pu,
     name: "U_PCC: cos phi = 1.0", line: {color: css("--muted"), width: 1.3, shape: "hv"}},
    {type: "scatter", mode: "lines", x: W.timestamps, y: W.modes[rcode].v_pcc_pu,
     name: `U_PCC: ${REG[rcode]}`, line: {color: css("--s1"), width: 2.0, shape: "hv"}},
    {type: "scatter", mode: "lines", x: W.timestamps, y: W.modes[rcode].q_pcc_mvar,
     name: "Q plant [Mvar] (right)", yaxis: "y2", line: {color: css("--s2"), width: 1.5, dash: "dot", shape: "hv"}},
  ];
  Plotly.react("plot-week", tracesWk, layoutBase({
    title: {text: `F1, ${fmt(W.p_inst_mw, 2)} MW PV, ${W.label}: ${REG[rcode]}`, font: {size: 13}},
    xaxis: axis({type: "date"}),
    yaxis: axis({title: {text: "PCC voltage [pu]"}}),
    yaxis2: axis({overlaying: "y", side: "right", showgrid: false, title: {text: "Q [Mvar]"}}),
    margin: {l: 56, r: 54, t: 32, b: 48},
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 1.05, y1: 1.05, line: {color: css("--bad"), dash: "dash", width: 1.2}}]
  }), cfg);
}

function renderStaticChartsAndTables() {
  const H = D.headline;
  const locs = ["F1", "F2", "F3", "GPZ_N1"];
  const narrowHc = window.innerWidth < 640;  // phone: three short lines per location, smaller ticks
  const sep = narrowHc ? "<br>" : " ";
  const locNames = [`F1${sep}overhead${sep}(14.2 km)`, `F2${sep}mixed${sep}(9.6 km)`, `F3${sep}cable${sep}(6.4 km)`, `GPZ${sep}busbar${sep}(N-1)`];
  const regOrder = [
    ["uncontrolled", "1. cos phi = 1.0", "#898781"],
    ["cos_phi_p", "2. cos phi(P)", "#e8a33d"],
    ["qu_droop", "3. Q(U) droop", "#2a78d6"],
  ];
  const hcBars = regOrder.map(([rc, rl, col]) => ({
    type: "bar", name: rl, x: locNames,
    y: locs.map(l => (D.hosting_table.find(r => r.feeder === l && r.regime === rc) || {}).hosting_capacity_mw || 0),
    marker: {color: col}
  }));
  Plotly.react("plot-hc", hcBars, layoutBase({
    barmode: "group",
    xaxis: axis({automargin: true, tickfont: {size: narrowHc ? 10 : 12}}),
    yaxis: axis({title: {text: "Connectable capacity [MW]"}}),
  }), cfg);

  const tHc = document.getElementById("tbl-hc");
  tHc.innerHTML = "<thead><tr><th>Location</th><th>Regime</th><th class='n'>Connectable [MW]</th><th class='n'>Max U [pu]</th><th class='n'>Max loading [%]</th><th class='n'>Plant Q [Mvar]</th><th class='n'>Loss [kW]</th><th>Binding constraint</th></tr></thead><tbody>" +
    D.hosting_table.map(r => `<tr><td>${r.feeder}</td><td>${REG[r.regime]}</td><td class='n'>${fmt(r.hosting_capacity_mw, 2)}</td><td class='n'>${fmt(r.v_pcc_pu, 4)}</td><td class='n'>${fmt(r.max_branch_loading_pct, 1)}</td><td class='n'>${fmt(r.q_net_mvar, 2)}</td><td class='n'>${fmt(r.feeder_loss_kw, 1)}</td><td>${r.binding_constraint}</td></tr>`).join("") +
    "</tbody>";

  const N1 = D.n1_study;
  document.getElementById("txt-curt").textContent =
    `A ${H.highlight_cap_pct} % export cap lets ${fmt(1 / (H.highlight_cap_pct / 100), 2)} MW of PV share 1 MW of export limit (arithmetic); ` +
    `over the PVGIS ${H.pv_year} year it curtails ${fmt(H.curt_hl_pct, 2)} % of that PV energy ` +
    `(${fmt(H.curt_hl_min_pct, 2)}-${fmt(H.curt_hl_max_pct, 2)} % over the PVGIS years 2005-2023) and delivers ` +
    `${fmt(H.energy_gain_hl_pct, 1)} % more MWh through the same limit than a firm plant. ` +
    `Right: hourly N-1 power flow of the GPZ for the firm, capped and pooled (PV + wind + battery) busbar scenarios; ` +
    `highest transformer loading ${fmt(N1.max_s_firm_pv_mva, 2)} / ${fmt(N1.max_s_capped_pv_mva, 2)} / ${fmt(N1.max_s_pooling_bess_mva, 2)} MVA ` +
    `against ${fmt(N1.s_limit_mva, 0)} MVA.`;

  const C = D.curtailment;
  Plotly.react("plot-curt", [
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.curtailed_energy_pct),
     name: "Curtailed PV energy [%]", line: {color: css("--s2"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.bess_lost_energy_pct),
     name: "Not delivered with battery [%]", line: {color: css("--s3"), width: 2}},
    {type: "scatter", mode: "lines+markers", x: C.map(r => r.export_cap_pct), y: C.map(r => r.energy_gain_vs_firm_pct),
     name: "MWh gain vs firm [%] (right)", yaxis: "y2", line: {color: css("--s1"), width: 2, dash: "dash"}},
  ], layoutBase({
    title: {text: "Export cap vs energy", font: {size: 13}},
    xaxis: axis({title: {text: "Export cap [% of P_inst]"}, autorange: "reversed"}),
    yaxis: axis({title: {text: "Energy not delivered [% of MWh]"}}),
    yaxis2: axis({overlaying: "y", side: "right", showgrid: false, title: {text: "MWh gain [%]"}}),
    margin: {l: 56, r: 54, t: 32, b: 48}
  }), cfg);

  Plotly.react("plot-n1", [
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_firm_pv,
     name: `A: ${fmt(N1.p_firm_mw, 2)} MW PV firm`, line: {color: css("--muted"), width: 2}},
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_capped_pv,
     name: `B: ${fmt(N1.capped_pv_mw, 2)} MW PV capped`, line: {color: css("--s3"), width: 2}},
    {type: "scatter", mode: "lines", x: N1.duration_hours, y: N1.s_mva_pooling_bess,
     name: `C: + ${fmt(N1.pooling_wind_mw, 2)} MW wind + BESS`, line: {color: css("--s1"), width: 2.2}},
  ], layoutBase({
    title: {text: "GPZ transformer S [MVA] under N-1, duration curve", font: {size: 13}},
    xaxis: axis({title: {text: "Hours per year [h]"}}),
    yaxis: axis({title: {text: "Apparent power S [MVA]"}, range: [0, 1.25 * N1.s_limit_mva]}),
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: N1.s_limit_mva, y1: N1.s_limit_mva, line: {color: css("--bad"), dash: "dash", width: 1.5}}]
  }), cfg);

  const V = D.verification;
  const vRows = [
    ["2-bus receiving-end voltage |V1| [pu]", V.v1_exact_pu.toFixed(10), V.v1_nr_pu.toFixed(10), V.v_abs_error_pu.toExponential(2)],
    ["2-bus receiving-end angle theta1 [rad]", V.theta1_exact_rad.toFixed(10), V.theta1_nr_rad.toFixed(10), V.theta_abs_error_rad.toExponential(2)],
    ["Series loss I²R [MW]", V.p_loss_exact_mw.toFixed(10), V.p_loss_nr_mw.toFixed(10), V.p_loss_error_mw.toExponential(2)],
    ["Active power balance [MW]", "0", V.p_balance_error_mw.toExponential(2), V.p_balance_error_mw.toExponential(2)],
    ["Reactive power balance [Mvar]", "0", V.q_balance_error_mvar.toExponential(2), V.q_balance_error_mvar.toExponential(2)],
  ];
  document.getElementById("tbl-ver").innerHTML =
    "<thead><tr><th>Quantity</th><th class='n'>Closed form</th><th class='n'>Newton-Raphson Ybus</th><th class='n'>Absolute error</th></tr></thead><tbody>" +
    vRows.map(r => `<tr><td>${r[0]}</td><td class='n'>${r[1]}</td><td class='n'>${r[2]}</td><td class='n'>${r[3]}</td></tr>`).join("") +
    "</tbody>";
}

function renderAll() {
  renderHeadlineTiles();
  updateInteractiveSim();
  renderStaticChartsAndTables();
}

initControls();
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

    def _hc_val(f: str, r: str) -> float:
        return float(df_hc[(df_hc["feeder"] == f) & (df_hc["regime"] == r)]["hosting_capacity_mw"].iloc[0])

    df_curt = compute_curtailment_tradeoff(df_profiles)
    df_curt.to_csv(results_dir / "curtailment_tradeoff.csv", index=False)
    df_cap_years = compute_export_cap_by_pv_year(HIGHLIGHT_CAP_PCT)
    df_cap_years.to_csv(results_dir / "export_cap_by_pv_year.csv", index=False)

    # Reactive study plant: the F1 unity-power-factor hosting capacity, feasible under all three regimes
    loss_cmp = compute_annual_reactive_loss_comparison(
        net_n0, df_profiles, feeder_code="F1", p_inst_mw=_hc_val("F1", "uncontrolled")
    )
    n1_study = compute_n1_transformer_study(
        net_n1, df_profiles, p_firm_mw=_hc_val("GPZ_N1", "uncontrolled"), export_cap_share=HIGHLIGHT_CAP_PCT / 100.0
    )
    week_trace = simulate_sunny_week_trace(net_n0, df_profiles, feeder_code="F1", p_inst_mw=_hc_val("F1", "qu_droop"))
    spatial_profiles = compute_spatial_voltage_profiles(net_n1)
    ver = verify_two_bus_analytical()
    (results_dir / "verification.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")

    # Figure 1 right panel: largest slider capacity not above the F1 Q(U) hosting capacity
    spatial_ref_mw = f"{max(c for c in SPATIAL_CAPACITIES_MW if c <= _hc_val('F1', 'qu_droop')):.1f}"

    row_hl = df_curt[df_curt["export_cap_pct"] == HIGHLIGHT_CAP_PCT].iloc[0]
    headline = {
        "f1_uncontrolled_mw": _hc_val("F1", "uncontrolled"),
        "f1_cosphi_mw": _hc_val("F1", "cos_phi_p"),
        "f1_qu_mw": _hc_val("F1", "qu_droop"),
        "f2_uncontrolled_mw": _hc_val("F2", "uncontrolled"),
        "f2_cosphi_mw": _hc_val("F2", "cos_phi_p"),
        "f2_qu_mw": _hc_val("F2", "qu_droop"),
        "f3_uncontrolled_mw": _hc_val("F3", "uncontrolled"),
        "f3_cosphi_mw": _hc_val("F3", "cos_phi_p"),
        "f3_qu_mw": _hc_val("F3", "qu_droop"),
        "gpz_n1_uncontrolled_mw": _hc_val("GPZ_N1", "uncontrolled"),
        "gpz_n1_cosphi_mw": _hc_val("GPZ_N1", "cos_phi_p"),
        "gpz_n1_qu_mw": _hc_val("GPZ_N1", "qu_droop"),
        "f1_qu_gain_pct": round(100.0 * (_hc_val("F1", "qu_droop") / _hc_val("F1", "uncontrolled") - 1.0), 1),
        "f1_cosphi_gain_pct": round(100.0 * (_hc_val("F1", "cos_phi_p") / _hc_val("F1", "uncontrolled") - 1.0), 1),
        "f3_cosphi_change_pct": round(100.0 * (_hc_val("F3", "cos_phi_p") / _hc_val("F3", "uncontrolled") - 1.0), 1),
        "f3_qu_change_pct": round(100.0 * (_hc_val("F3", "qu_droop") / _hc_val("F3", "uncontrolled") - 1.0), 1),
        "reactive_study_p_mw": loss_cmp["p_inst_mw"],
        "q_reduction_pct": loss_cmp["q_reduction_pct"],
        "extra_loss_reduction_pct": loss_cmp["extra_loss_reduction_pct"],
        "highlight_cap_pct": HIGHLIGHT_CAP_PCT,
        "curt_hl_pct": float(row_hl["curtailed_energy_pct"]),
        "curt_hl_hours": int(row_hl["curtailed_hours_per_yr"]),
        "energy_gain_hl_pct": float(row_hl["energy_gain_vs_firm_pct"]),
        "bess_lost_hl_pct": float(row_hl["bess_lost_energy_pct"]),
        "curt_hl_min_pct": float(df_cap_years["curtailed_energy_pct"].min()),
        "curt_hl_median_pct": float(df_cap_years["curtailed_energy_pct"].median()),
        "curt_hl_max_pct": float(df_cap_years["curtailed_energy_pct"].max()),
        "pv_year": PV_YEAR,
        "pv_dc_yield": round(float(pv_dc_year(PV_YEAR).sum()), 1),
        "pv_dc_yield_mean": round(float(df_cap_years["dc_yield_kwh_per_kwp"].mean()), 1),
        "pooling_spilled_nobess_pct": n1_study["pooling_spilled_nobess_pct"],
        "pooling_lost_bess_pct": n1_study["pooling_lost_bess_pct"],
        "pv_flh": round(float(df_profiles["pv_cf"].sum()), 1),
        "pv_peak_pu": round(float(df_profiles["pv_cf"].max()), 3),
        "wind_flh": round(float(df_profiles["wind_cf"].sum()), 1),
        "gpz_existing_gen_mw": GPZ_EXISTING_GEN_MW,
    }

    summary_payload = {
        "headline": headline,
        "network": describe_network(net_n1),
        "hosting_table": df_hc.to_dict(orient="records"),
        "curtailment": df_curt.to_dict(orient="records"),
        "export_cap_by_pv_year": df_cap_years.to_dict(orient="records"),
        "reactive_loss_comparison": loss_cmp,
        "n1_study": n1_study,
        "week_trace": week_trace,
        "spatial_capacities_mw": SPATIAL_CAPACITIES_MW,
        "spatial": spatial_profiles,
        "verification": ver,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    generate_plots(df_hc, spatial_profiles, spatial_ref_mw, week_trace, df_curt, n1_study, img_dir)
    generate_dashboard_html(summary_payload, docs_dir)
    return summary_payload
