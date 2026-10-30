"""NC RfG reactive power control modes at the 15 kV PCC: unity power factor, cos-phi(P) and Q(U) droop.

The three modes belong to the families listed in Commission Regulation (EU) 2016/631 (NC RfG) Art. 21(3)(d)
and EN 50549-2 (reactive power setpoint, power factor, cos-phi(P), Q(U)). The parameter values used here
(Qmax = 0.33 Pn, i.e. cos phi = 0.95 at rated power; Q(U) deadband 0.98-1.02 pu, saturation at 0.95/1.05 pu;
cos-phi(P) knee at 50 % Pn) are illustrative study assumptions. The actual settings are set by the Polish
requirements of general application under NC RfG and by the DSO (IRiESD and the connection conditions);
they change over time and must be verified for any real project.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .network import NetworkModel
from .powerflow import PowerFlowResult, QOfV, solve_ac_powerflow


QU_Q_MAX_RATIO: float = 0.33          # Qmax / Pn of the Q(U) characteristic (cos phi = 0.95 equivalent)
QU_U_SAT_LOW: float = 0.95
QU_U_DB_LOW: float = 0.98
QU_U_DB_HIGH: float = 1.02
QU_U_SAT_HIGH: float = 1.05
QU_MIN_P_SHARE: float = 0.10          # below 10 % Pn the Q(U) capability is reduced proportionally


@dataclass(frozen=True)
class ControlRegimeSpec:
    """Specification of a reactive power control regime of the connected power park module."""

    code: str
    name: str
    short_label: str
    uses_qu_droop: bool
    uses_cos_phi_p: bool


CONTROL_REGIMES: dict[str, ControlRegimeSpec] = {
    "uncontrolled": ControlRegimeSpec(
        code="uncontrolled",
        name="1. Unity power factor (cos phi = 1.0)",
        short_label="cos phi = 1.0",
        uses_qu_droop=False,
        uses_cos_phi_p=False,
    ),
    "cos_phi_p": ControlRegimeSpec(
        code="cos_phi_p",
        name="2. cos phi(P): 1.0 up to 50 % Pn, linear to 0.95 ind at Pn",
        short_label="cos phi(P) 0.95 ind",
        uses_qu_droop=False,
        uses_cos_phi_p=True,
    ),
    "qu_droop": ControlRegimeSpec(
        code="qu_droop",
        name="3. Q(U) droop (deadband 0.98-1.02 pu, Qmax = 0.33 Pn at 1.05 pu)",
        short_label="Q(U) droop",
        uses_qu_droop=True,
        uses_cos_phi_p=False,
    ),
}


def compute_cos_phi_p_q_mvar(
    p_export_mw: float | np.ndarray,
    p_nom_mw: float,
    cos_phi_min: float = 0.95,
    p_knee_share: float = 0.50,
) -> float | np.ndarray:
    """Reactive power Q [Mvar] at the 15 kV PCC under a cos-phi(P) characteristic (negative = absorbed).

    cos phi = 1.0 up to p_knee_share * Pn, then decreasing linearly with P to cos_phi_min (inductive,
    under-excited) at P = Pn: Q = -P tan(arccos(cos phi(P))).
    """
    p_arr = np.asarray(p_export_mw, dtype=float)
    if p_nom_mw <= 1e-9:
        return 0.0 if np.ndim(p_export_mw) == 0 else np.zeros_like(p_arr)

    loading = np.clip(p_arr / p_nom_mw, 0.0, 1.0)
    frac = np.clip((loading - p_knee_share) / (1.0 - p_knee_share), 0.0, 1.0)
    cos_phi = 1.0 - (1.0 - cos_phi_min) * frac
    q_pcc_mvar = -p_arr * np.tan(np.arccos(cos_phi))
    return float(q_pcc_mvar) if np.ndim(p_export_mw) == 0 else q_pcc_mvar


def qu_droop_q_and_slope(
    v_pu: np.ndarray,
    q_lim_mvar: float | np.ndarray,
    u_sat_low: float = QU_U_SAT_LOW,
    u_db_low: float = QU_U_DB_LOW,
    u_db_high: float = QU_U_DB_HIGH,
    u_sat_high: float = QU_U_SAT_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear Q(U) characteristic: reactive power [Mvar] and its slope dQ/dU [Mvar/pu]."""
    v = np.asarray(v_pu, dtype=float)
    q_lim = np.broadcast_to(np.asarray(q_lim_mvar, dtype=float), v.shape)
    span_hi = u_sat_high - u_db_high
    span_lo = u_db_low - u_sat_low

    hi = v > u_db_high
    lo = v < u_db_low
    q = np.where(hi, -q_lim * np.clip((v - u_db_high) / span_hi, 0.0, 1.0), 0.0)
    q = np.where(lo, q_lim * np.clip((u_db_low - v) / span_lo, 0.0, 1.0), q)
    dq = np.where(hi & (v < u_sat_high), -q_lim / span_hi, 0.0)
    dq = np.where(lo & (v > u_sat_low), -q_lim / span_lo, dq)
    return q, dq


def qu_q_limit_mvar(
    p_nom_mw: float | np.ndarray,
    p_avail_mw: float | np.ndarray | None = None,
    q_max_ratio: float = QU_Q_MAX_RATIO,
) -> float | np.ndarray:
    """Q(U) saturation reactive power [Mvar]: q_max_ratio * Pn, reduced proportionally below 10 % Pn."""
    p_nom = np.asarray(p_nom_mw, dtype=float)
    q_lim = q_max_ratio * p_nom
    if p_avail_mw is not None:
        p_av = np.asarray(p_avail_mw, dtype=float)
        q_lim = q_lim * np.clip(p_av / np.maximum(QU_MIN_P_SHARE * p_nom, 1e-9), 0.0, 1.0)
    return float(q_lim) if np.ndim(q_lim) == 0 else q_lim


def compute_qu_droop_q_mvar(
    v_pcc_pu: float | np.ndarray,
    p_nom_mw: float,
    p_avail_mw: float | np.ndarray | None = None,
    u_db_low: float = QU_U_DB_LOW,
    u_db_high: float = QU_U_DB_HIGH,
    u_sat_low: float = QU_U_SAT_LOW,
    u_sat_high: float = QU_U_SAT_HIGH,
    q_max_ratio: float = QU_Q_MAX_RATIO,
) -> float | np.ndarray:
    """Compute the autonomous Q(U) droop reactive power [Mvar] at the 15 kV PCC.

    Piecewise-linear characteristic with deadband [u_db_low, u_db_high] = [0.98, 1.02] pu:
      - U in [0.98, 1.02] pu: Q = 0 Mvar (no reactive circulation at normal voltages)
      - U > 1.02 pu: linear inductive absorption down to Q = -0.33 * Pn at U >= 1.05 pu
      - U < 0.98 pu: linear capacitive injection up to Q = +0.33 * Pn at U <= 0.95 pu
    When p_avail_mw is given and below 10 % Pn, the reactive capability is reduced proportionally.
    """
    q_lim = qu_q_limit_mvar(p_nom_mw, p_avail_mw, q_max_ratio)
    q, _ = qu_droop_q_and_slope(v_pcc_pu, q_lim, u_sat_low, u_db_low, u_db_high, u_sat_high)
    return float(q) if np.ndim(v_pcc_pu) == 0 else q


def make_qu_law(q_lim_mvar: np.ndarray) -> QOfV:
    """Q(U) law for solve_ac_powerflow(q_of_v=...) with a per-bus saturation reactive power [Mvar]."""
    q_lim = np.asarray(q_lim_mvar, dtype=float)
    return lambda v: qu_droop_q_and_slope(v, q_lim)


def solve_feeder_with_ncrfg(
    net: NetworkModel,
    feeder_code: str,
    p_inst_mw: float,
    regime_code: str = "qu_droop",
    cf_avail: float = 1.0,
    load_scale: float | None = None,
) -> PowerFlowResult:
    """Solve the AC power flow with a candidate plant on a feeder under one reactive control regime.

    The candidate generation sits on the distal corridor of the selected feeder (75 % at the terminal PCC bus,
    25 % at the adjacent upstream node). Loads are the minimum daytime loads, or the peak loads times
    load_scale when given. Under Q(U) the droop law is solved inside the Newton-Raphson iteration.
    """
    reg = CONTROL_REGIMES[regime_code]
    fspec = net.feeders[feeder_code]
    n = len(net.buses)

    p_export_mw = p_inst_mw * float(cf_avail)

    p_gen = np.zeros(n, dtype=float)
    q_gen = np.zeros(n, dtype=float)

    pcc_bus = fspec.pcc_bus_id
    adj_bus = fspec.bus_ids[-2] if len(fspec.bus_ids) >= 2 else pcc_bus
    w_pcc, w_adj = (0.75, 0.25) if adj_bus != pcc_bus else (1.0, 0.0)
    p_gen[pcc_bus] = w_pcc * p_export_mw
    p_gen[adj_bus] = w_adj * p_export_mw

    if load_scale is not None:
        p_load = np.array([b.p_load_peak_mw * load_scale for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_peak_mvar * load_scale for b in net.buses], dtype=float)
    else:
        p_load = np.array([b.p_load_min_mw for b in net.buses], dtype=float)
        q_load = np.array([b.q_load_min_mvar for b in net.buses], dtype=float)

    if reg.uses_cos_phi_p:
        q_tot = float(compute_cos_phi_p_q_mvar(p_export_mw, p_nom_mw=p_inst_mw))
        q_gen[pcc_bus] = w_pcc * q_tot
        q_gen[adj_bus] = w_adj * q_tot

    if reg.uses_qu_droop and p_inst_mw > 1e-6:
        q_lim_total = qu_q_limit_mvar(p_inst_mw, p_export_mw)
        if adj_bus != pcc_bus:
            qv_buses, q_lims = np.array([pcc_bus, adj_bus]), np.array([w_pcc, w_adj]) * q_lim_total
        else:
            qv_buses, q_lims = np.array([pcc_bus]), np.array([q_lim_total])
        return solve_ac_powerflow(
            net,
            p_gen_mw=p_gen,
            q_gen_mvar=q_gen,
            p_load_mw=p_load,
            q_load_mvar=q_load,
            q_of_v_buses=qv_buses,
            q_of_v=make_qu_law(q_lims),
        )

    return solve_ac_powerflow(
        net,
        p_gen_mw=p_gen,
        q_gen_mvar=q_gen,
        p_load_mw=p_load,
        q_load_mvar=q_load,
    )
