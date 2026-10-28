"""NC RfG Type B/C reactive power controllers: cos-phi(P), Q(U) droop, export cap, and BESS cable pooling."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .network import NetworkModel
from .powerflow import PowerFlowResult, solve_ac_powerflow


@dataclass(frozen=True)
class ControlRegimeSpec:
    """Specification of a grid connection and reactive power control regime."""

    code: str
    name: str
    short_label: str
    export_cap_share: float
    bess_peak_shave_share: float
    uses_qu_droop: bool
    uses_cos_phi_p: bool


CONTROL_REGIMES: dict[str, ControlRegimeSpec] = {
    "uncontrolled": ControlRegimeSpec(
        code="uncontrolled",
        name="1. Uncontrolled (cos phi = 1.0, firm 100 % export)",
        short_label="cos phi = 1.0 (firm)",
        export_cap_share=1.00,
        bess_peak_shave_share=1.00,
        uses_qu_droop=False,
        uses_cos_phi_p=False,
    ),
    "cos_phi_fixed": ControlRegimeSpec(
        code="cos_phi_fixed",
        name="2. Fixed inductive PF (cos phi = 0.95 ind above 50 % Pn)",
        short_label="Fixed cos phi = 0.95 ind",
        export_cap_share=1.00,
        bess_peak_shave_share=1.00,
        uses_qu_droop=False,
        uses_cos_phi_p=True,
    ),
    "qu_droop": ControlRegimeSpec(
        code="qu_droop",
        name="3. NC RfG Q(U) droop (deadband 0.98-1.02 pu, Qmax = +-0.33 Pn)",
        short_label="NC RfG Q(U) droop",
        export_cap_share=1.00,
        bess_peak_shave_share=1.00,
        uses_qu_droop=True,
        uses_cos_phi_p=False,
    ),
    "non_firm_70": ControlRegimeSpec(
        code="non_firm_70",
        name="4. Non-firm connection (70 % export cap + NC RfG Q(U))",
        short_label="Q(U) + 70 % export cap",
        export_cap_share=0.70,
        bess_peak_shave_share=0.70,
        uses_qu_droop=True,
        uses_cos_phi_p=False,
    ),
    "cable_pooling_bess": ControlRegimeSpec(
        code="cable_pooling_bess",
        name="5. Cable pooling + co-located 2 h BESS (peak-shaving + Q(U))",
        short_label="Q(U) + 2 h BESS pooling",
        export_cap_share=0.5945,
        bess_peak_shave_share=0.5945,
        uses_qu_droop=True,
        uses_cos_phi_p=False,
    ),
}


def compute_cos_phi_p_q_mvar(
    p_export_mw: float | np.ndarray,
    p_nom_mw: float,
    cos_phi_target: float = 0.95,
    p_threshold_share: float = 0.30,
    step_up_tr_uk_pct: float = 2.8,
) -> float | np.ndarray:
    """Compute net 15 kV PCC reactive power Q [Mvar] under fixed cos-phi(P) inductive control.

    Per Polish DSO Type B/C settings, inductive reactive absorption activates above 50 % Pn
    at the generator LV terminals (tan phi_inv = tan(arccos(0.95)) = 0.32868) and experiences
    reactive loss I^2 X_tr across the plant's 0.8/15 kV unit transformer (u_k = 6.5 %).
    """
    p_arr = np.asarray(p_export_mw, dtype=float)
    if p_nom_mw <= 1e-9:
        return 0.0 if np.ndim(p_export_mw) == 0 else np.zeros_like(p_arr)

    tan_max = float(np.tan(np.arccos(cos_phi_target)))  # 0.328684 at cos_phi = 0.95
    loading = np.clip(p_arr / p_nom_mw, 0.0, 1.0)

    # Active above 50 % Pn (with linear ramp from 0.20 Pn to 0.50 Pn, or standard 0.50 Pn knee)
    ramp = np.where(
        loading >= p_threshold_share,
        1.0,
        np.clip((loading - 0.20) / max(1e-6, p_threshold_share - 0.20), 0.0, 1.0),
    )
    q_inv_mvar = -p_arr * tan_max * ramp
    # Subtract step-up transformer (0.8/15 kV) series reactive power loss so net Q at 15 kV PCC is exact
    s_sq_ratio = (p_arr ** 2 + q_inv_mvar ** 2) / max(1e-6, p_nom_mw ** 2)
    q_tr_loss_mvar = (step_up_tr_uk_pct / 100.0) * p_nom_mw * s_sq_ratio
    q_pcc_mvar = np.minimum(0.0, q_inv_mvar + q_tr_loss_mvar)
    return float(q_pcc_mvar) if np.ndim(p_export_mw) == 0 else q_pcc_mvar


def compute_qu_droop_q_mvar(
    v_pcc_pu: float | np.ndarray,
    p_nom_mw: float,
    p_avail_mw: float | np.ndarray | None = None,
    u_db_low: float = 0.98,
    u_db_high: float = 1.02,
    u_sat_low: float = 0.95,
    u_sat_high: float = 1.05,
    q_max_ratio: float = 0.415,
) -> float | np.ndarray:
    """Compute NC RfG Type B/C autonomous Q(U) droop reactive power [Mvar] at the 15 kV PCC.

    Piecewise-linear characteristic with deadband [u_db_low, u_db_high] = [0.98, 1.02] pu:
      - U in [0.98, 1.02] pu: Q = 0 Mvar (eliminates reactive circulation during normal voltages)
      - U > 1.02 pu: linear inductive absorption up to Q = -0.33 * P_ref at U >= 1.05 pu
      - U < 0.98 pu: linear capacitive injection up to Q = +0.33 * P_ref at U <= 0.95 pu
    When p_avail_mw is provided and below 10 % Pn, reactive capability is gated proportionally.
    """
    v_arr = np.asarray(v_pcc_pu, dtype=float)
    q_lim = q_max_ratio * p_nom_mw

    if p_avail_mw is not None:
        p_av = np.asarray(p_avail_mw, dtype=float)
        gate = np.clip(p_av / max(1e-6, 0.15 * p_nom_mw), 0.0, 1.0)
        q_lim = q_lim * gate

    q_out = np.zeros_like(v_arr, dtype=float)
    high_mask = v_arr > u_db_high
    low_mask = v_arr < u_db_low

    if np.any(high_mask):
        frac_hi = np.clip((v_arr - u_db_high) / (u_sat_high - u_db_high), 0.0, 1.0)
        q_out = np.where(high_mask, -q_lim * frac_hi, q_out)

    if np.any(low_mask):
        frac_lo = np.clip((u_db_low - v_arr) / (u_db_low - u_sat_low), 0.0, 1.0)
        q_out = np.where(low_mask, +q_lim * frac_lo, q_out)

    return float(q_out) if np.ndim(v_pcc_pu) == 0 else q_out


def solve_feeder_with_ncrfg(
    net: NetworkModel,
    feeder_code: str,
    p_inst_mw: float,
    regime_code: str = "qu_droop",
    cf_avail: float = 1.0,
    load_scale: float | None = None,
    p_load_override_mw: np.ndarray | None = None,
    q_load_override_mvar: np.ndarray | None = None,
    max_outer_iter: int = 12,
    outer_tol_pu: float = 1e-6,
) -> PowerFlowResult:
    """Solve coupled AC power flow + NC RfG reactive control on a target feeder.

    Distributes candidate generation along the distal corridor of the selected feeder
    (75 % at the terminal PCC bus, 25 % at the adjacent upstream collector node) and
    iterates the Q(U) droop law to a damped fixed-point equilibrium.
    """
    reg = CONTROL_REGIMES[regime_code]
    fspec = net.feeders[feeder_code]
    n = len(net.buses)

    p_raw_mw = p_inst_mw * float(cf_avail)
    p_export_mw = min(p_raw_mw, p_inst_mw * reg.export_cap_share)

    p_gen = np.zeros(n, dtype=float)
    q_gen = np.zeros(n, dtype=float)

    # Candidate RES connects primarily at the feeder PCC (75 %) and adjacent collector node (25 %)
    pcc_bus = fspec.pcc_bus_id
    adj_bus = fspec.bus_ids[-2] if len(fspec.bus_ids) >= 2 else pcc_bus
    w_pcc, w_adj = (0.75, 0.25) if adj_bus != pcc_bus else (1.0, 0.0)
    p_gen[pcc_bus] = w_pcc * p_export_mw
    p_gen[adj_bus] = w_adj * p_export_mw

    if p_load_override_mw is not None and q_load_override_mvar is not None:
        p_load = p_load_override_mw
        q_load = q_load_override_mvar
    elif load_scale is not None:
        p_load = np.array([b.p_load_peak_mw * load_scale for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_peak_mvar * load_scale for b in net.buses], dtype=float)
    else:
        p_load = np.array([b.p_load_min_mw for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_min_mvar for b in net.buses], dtype=float)

    if reg.uses_cos_phi_p:
        q_tot = float(compute_cos_phi_p_q_mvar(p_export_mw, p_nom_mw=max(p_inst_mw * reg.export_cap_share, 1e-6)))
        q_gen[pcc_bus] = w_pcc * q_tot
        q_gen[adj_bus] = w_adj * q_tot
        return solve_ac_powerflow(
            net,
            p_gen_mw=p_gen,
            q_gen_mvar=q_gen,
            p_load_mw=p_load,
            q_load_mvar=q_load,
        )

    if not reg.uses_qu_droop or p_inst_mw <= 1e-6:
        return solve_ac_powerflow(
            net,
            p_gen_mw=p_gen,
            q_gen_mvar=q_gen,
            p_load_mw=p_load,
            q_load_mvar=q_load,
        )

    # Damped fixed-point iteration for autonomous Q(U) voltage droop
    res = solve_ac_powerflow(
        net,
        p_gen_mw=p_gen,
        q_gen_mvar=q_gen,
        p_load_mw=p_load,
        q_load_mvar=q_load,
    )
    p_ref_mw = p_inst_mw * reg.export_cap_share
    alpha = 0.45
    for _ in range(max_outer_iter):
        v_pcc = float(res.v_mag_pu[pcc_bus])
        v_adj = float(res.v_mag_pu[adj_bus])
        q_target_pcc = float(
            compute_qu_droop_q_mvar(v_pcc, p_nom_mw=w_pcc * p_ref_mw, p_avail_mw=w_pcc * p_export_mw)
        )
        q_target_adj = float(
            compute_qu_droop_q_mvar(v_adj, p_nom_mw=w_adj * p_ref_mw, p_avail_mw=w_adj * p_export_mw)
        )
        dq = max(abs(q_target_pcc - q_gen[pcc_bus]), abs(q_target_adj - q_gen[adj_bus]))
        q_gen[pcc_bus] = (1.0 - alpha) * q_gen[pcc_bus] + alpha * q_target_pcc
        q_gen[adj_bus] = (1.0 - alpha) * q_gen[adj_bus] + alpha * q_target_adj
        res_new = solve_ac_powerflow(
            net,
            p_gen_mw=p_gen,
            q_gen_mvar=q_gen,
            p_load_mw=p_load,
            q_load_mvar=q_load,
        )
        dv = float(np.max(np.abs(res_new.v_mag_pu - res.v_mag_pu)))
        res = res_new
        if dv < outer_tol_pu and dq < 1e-6:
            break

    return res
