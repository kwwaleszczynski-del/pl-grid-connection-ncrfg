"""Exact Newton-Raphson AC power flow solver (Ybus) and 2-bus analytical verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .network import BASE_MVA, BASE_KV_MV, NetworkModel, build_Ybus


@dataclass(frozen=True)
class PowerFlowResult:
    """Steady-state AC power flow solution on the Polish GPZ + 15 kV network."""

    converged: bool
    iterations: int
    max_mismatch_pu: float
    v_mag_pu: np.ndarray
    v_ang_rad: np.ndarray
    p_inj_mw: np.ndarray
    q_inj_mvar: np.ndarray
    p_slack_mw: float
    q_slack_mvar: float
    total_p_loss_mw: float
    total_q_loss_mvar: float
    branch_p_from_mw: np.ndarray
    branch_q_from_mvar: np.ndarray
    branch_s_max_end_mva: np.ndarray
    branch_i_ka: np.ndarray
    branch_loading_pct: np.ndarray
    branch_p_loss_mw: np.ndarray
    branch_q_loss_mvar: np.ndarray
    tr_s_mva: float
    tr_loading_n0_pct: float
    tr_loading_n1_pct: float
    oltc_tap_ratio: float
    q_v_mvar: np.ndarray


QOfV = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def solve_ac_powerflow(
    net: NetworkModel,
    p_gen_mw: np.ndarray | None = None,
    q_gen_mvar: np.ndarray | None = None,
    p_load_mw: np.ndarray | None = None,
    q_load_mvar: np.ndarray | None = None,
    v_slack_pu: float = 1.000,
    regulate_gpz_15kv: bool = True,
    tol: float = 1e-11,
    max_iter: int = 30,
    q_of_v_buses: np.ndarray | None = None,
    q_of_v: QOfV | None = None,
) -> PowerFlowResult:
    """Solve exact polar Newton-Raphson AC power flow using Ybus.

    Parameters
    ----------
    net : NetworkModel
        Polish GPZ 110/15 kV + 4 feeders network model.
    p_gen_mw, q_gen_mvar : np.ndarray, optional
        Active [MW] and reactive [Mvar] generation at each bus (+ into grid).
    p_load_mw, q_load_mvar : np.ndarray, optional
        Active [MW] and reactive [Mvar] load at each bus (+ consumed from grid).
        Defaults to minimum daytime load of each bus.
    v_slack_pu : float
        Voltage magnitude at Slack Bus 0 (110 kV grid), default 1.000 pu.
    regulate_gpz_15kv : bool
        If True, selects the OLTC tap ratio so that the 15 kV GPZ busbar (Bus 1)
        is regulated to its nominal setpoint (1.000 pu) prior to feeder evaluation.
    tol : float
        Convergence tolerance on maximum P and Q mismatch in per-unit on BASE_MVA.
    max_iter : int
        Maximum Newton-Raphson iterations.
    q_of_v_buses, q_of_v : optional
        Voltage-dependent reactive injection (e.g. a Q(U) droop) at the listed buses:
        q_of_v(v_pu) returns (Q [Mvar], dQ/dV [Mvar/pu]) for those buses. It is added to
        q_gen_mvar inside the Newton iteration (its slope enters the Jacobian), so the
        controller equilibrium is solved together with the power flow.
    """
    n = len(net.buses)
    p_g = np.zeros(n, dtype=float) if p_gen_mw is None else np.asarray(p_gen_mw, dtype=float)
    q_g = np.zeros(n, dtype=float) if q_gen_mvar is None else np.asarray(q_gen_mvar, dtype=float)

    if p_load_mw is None:
        p_l = np.array([b.p_load_min_mw for b in net.buses], dtype=float)
    else:
        p_l = np.asarray(p_load_mw, dtype=float)

    if q_load_mvar is None:
        q_l = np.array([b.q_load_min_mvar for b in net.buses], dtype=float)
    else:
        q_l = np.asarray(q_load_mvar, dtype=float)

    p_spec_pu = (p_g - p_l) / BASE_MVA
    q_spec_pu = (q_g - q_l) / BASE_MVA

    slack_idx = 0
    # All non-slack buses are PQ buses (indices 1..n-1), so the PQ sub-blocks are plain slices
    n_pq = n - 1
    qv_idx = np.asarray(q_of_v_buses, dtype=int) if (q_of_v is not None and q_of_v_buses is not None) else None

    def _q_of_v_pu(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_mvar, dq_dv_mvar = q_of_v(v[qv_idx])
        return np.asarray(q_mvar, dtype=float) / BASE_MVA, np.asarray(dq_dv_mvar, dtype=float) / BASE_MVA

    def _run_nr(Ybus: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
        G = Ybus.real
        B = Ybus.imag
        diag_G = np.diag(G)
        diag_B = np.diag(B)
        v = np.ones(n, dtype=float)
        v[slack_idx] = v_slack_pu
        theta = np.zeros(n, dtype=float)
        J = np.empty((2 * n_pq, 2 * n_pq), dtype=float)

        converged = False
        max_mis = 0.0
        it = 0
        for it in range(1, max_iter + 1):
            dtheta = theta[:, None] - theta[None, :]
            cos_dt = np.cos(dtheta)
            sin_dt = np.sin(dtheta)
            Gcos_Bsin = G * cos_dt + B * sin_dt
            Gsin_Bcos = G * sin_dt - B * cos_dt

            p_calc = v * (Gcos_Bsin @ v)
            q_calc = v * (Gsin_Bcos @ v)

            q_spec_it = q_spec_pu
            if qv_idx is not None:
                q_v_pu, dq_dv_pu = _q_of_v_pu(v)
                q_spec_it = q_spec_pu.copy()
                q_spec_it[qv_idx] += q_v_pu

            mismatch = np.concatenate([p_spec_pu[1:] - p_calc[1:], q_spec_it[1:] - q_calc[1:]])
            max_mis = float(np.max(np.abs(mismatch)))
            if max_mis < tol:
                converged = True
                break

            # Analytical Polar Jacobian J = [[H, N], [M, L]]
            vv = np.outer(v, v)
            H = vv * Gsin_Bcos
            np.fill_diagonal(H, -q_calc - (v ** 2) * diag_B)

            N = (v[:, None] * Gcos_Bsin)
            np.fill_diagonal(N, p_calc / v + v * diag_G)

            M = -vv * Gcos_Bsin
            np.fill_diagonal(M, p_calc - (v ** 2) * diag_G)

            L = (v[:, None] * Gsin_Bcos)
            np.fill_diagonal(L, q_calc / v - v * diag_B)
            if qv_idx is not None:
                # Specified Q depends on V: d(Q_calc - Q_spec)/dV adds -dQ_spec/dV on the diagonal
                L[qv_idx, qv_idx] -= dq_dv_pu

            J[:n_pq, :n_pq] = H[1:, 1:]
            J[:n_pq, n_pq:] = N[1:, 1:]
            J[n_pq:, :n_pq] = M[1:, 1:]
            J[n_pq:, n_pq:] = L[1:, 1:]

            dx = np.linalg.solve(J, mismatch)
            theta[1:] += dx[:n_pq]
            v[1:] += dx[n_pq:]
            v[1:] = np.clip(v[1:], 0.5, 1.5)

        return v, theta, converged, it, max_mis

    tap = net.oltc_tap_ratio
    Ybus = build_Ybus(net, tap_override=tap)
    v, theta, converged, it, max_mis = _run_nr(Ybus)

    if regulate_gpz_15kv and n > 1:
        # Continuous OLTC model: rescale the off-nominal tap ratio until the GPZ 15 kV busbar sits at
        # v_target_mv_pu (two corrections bring it within ~1e-6 pu; tap range +-9 x 1.78 %)
        for _ in range(3):
            if abs(v[1] - net.substation.v_target_mv_pu) <= 1e-5:
                break
            tap = float(np.clip(tap * (v[1] / net.substation.v_target_mv_pu), 1.0 - 0.1602, 1.0 + 0.1602))
            Ybus = build_Ybus(net, tap_override=tap)
            v, theta, converged, it, max_mis = _run_nr(Ybus)

    v_complex = v * np.exp(1j * theta)
    i_inj_pu = Ybus @ v_complex
    s_inj_pu = v_complex * np.conj(i_inj_pu)
    p_inj_mw = s_inj_pu.real * BASE_MVA
    q_inj_mvar = s_inj_pu.imag * BASE_MVA

    # Compute exact branch flows and losses
    m = len(net.branches)
    br_p_from = np.zeros(m, dtype=float)
    br_q_from = np.zeros(m, dtype=float)
    br_s_max = np.zeros(m, dtype=float)
    br_i_ka = np.zeros(m, dtype=float)
    br_loading = np.zeros(m, dtype=float)
    br_p_loss = np.zeros(m, dtype=float)
    br_q_loss = np.zeros(m, dtype=float)

    for k, br in enumerate(net.branches):
        i = br.from_bus
        j = br.to_bus
        z = complex(br.r_pu, br.x_pu)
        y_ser = 1.0 / z
        y_sh = 1j * (br.b_pu / 2.0)
        t = tap if br.branch_id == 0 else br.tap_ratio

        vi = v_complex[i] / t
        vj = v_complex[j]
        i_from_pu = (vi - vj) * y_ser + vi * y_sh
        i_to_pu = (vj - vi) * y_ser + vj * y_sh

        s_from_mva = (v_complex[i] * np.conj(i_from_pu / t)) * BASE_MVA
        s_to_mva = (vj * np.conj(i_to_pu)) * BASE_MVA
        s_loss_mva = s_from_mva + s_to_mva

        br_p_from[k] = float(s_from_mva.real)
        br_q_from[k] = float(s_from_mva.imag)
        s_end_mva = max(abs(s_from_mva), abs(s_to_mva))
        br_s_max[k] = float(s_end_mva)

        i_mag_pu = max(abs(i_from_pu / t), abs(i_to_pu))
        i_ka = float(i_mag_pu * BASE_MVA / (np.sqrt(3.0) * BASE_KV_MV))
        br_i_ka[k] = i_ka
        br_loading[k] = float(100.0 * s_end_mva / br.s_max_mva)
        br_p_loss[k] = float(s_loss_mva.real)
        br_q_loss[k] = float(s_loss_mva.imag)

    # Add fixed transformer core iron losses (22 kW per in-service transformer)
    p_fe_total_mw = (net.n_transformers_in_service * net.substation.p_fe_kw) / 1000.0
    tr_s_mva = float(br_s_max[0]) if m > 0 else 0.0
    tr_n0_pct = 100.0 * tr_s_mva / net.substation.s_n0_mva
    tr_n1_pct = 100.0 * tr_s_mva / net.substation.s_n1_mva

    q_v_mvar = np.zeros(n, dtype=float)
    if qv_idx is not None:
        q_v_mvar[qv_idx] = np.asarray(q_of_v(v[qv_idx])[0], dtype=float)

    return PowerFlowResult(
        converged=converged,
        iterations=it,
        max_mismatch_pu=max_mis,
        v_mag_pu=v,
        v_ang_rad=theta,
        p_inj_mw=p_inj_mw,
        q_inj_mvar=q_inj_mvar,
        p_slack_mw=float(p_inj_mw[slack_idx]),
        q_slack_mvar=float(q_inj_mvar[slack_idx]),
        total_p_loss_mw=float(np.sum(br_p_loss) + p_fe_total_mw),
        total_q_loss_mvar=float(np.sum(br_q_loss)),
        branch_p_from_mw=br_p_from,
        branch_q_from_mvar=br_q_from,
        branch_s_max_end_mva=br_s_max,
        branch_i_ka=br_i_ka,
        branch_loading_pct=br_loading,
        branch_p_loss_mw=br_p_loss,
        branch_q_loss_mvar=br_q_loss,
        tr_s_mva=tr_s_mva,
        tr_loading_n0_pct=tr_n0_pct,
        tr_loading_n1_pct=tr_n1_pct,
        oltc_tap_ratio=tap,
        q_v_mvar=q_v_mvar,
    )


def solve_two_bus_analytical(
    v0_pu: float,
    r_pu: float,
    x_pu: float,
    p1_pu: float,
    q1_pu: float,
) -> tuple[float, float, float, float]:
    """Closed-form exact solution of the 2-bus AC power flow equation.

    Bus 0 is slack at V0_pu < 0 rad. Bus 1 injects net complex power S1 = P1 + j Q1 (per unit)
    through series impedance Z = R + j X.

    Returns (v1_pu, theta1_rad, p_loss_pu, q_loss_pu).
    """
    b = (v0_pu ** 2) + 2.0 * (r_pu * p1_pu + x_pu * q1_pu)
    c = (r_pu ** 2 + x_pu ** 2) * (p1_pu ** 2 + q1_pu ** 2)
    disc = b ** 2 - 4.0 * c
    if disc < 0.0:
        raise ValueError("2-bus analytical power flow is beyond voltage collapse limit.")
    v1_sq = 0.5 * (b + np.sqrt(disc))
    v1_pu = float(np.sqrt(v1_sq))
    theta1_rad = float(np.arctan2(x_pu * p1_pu - r_pu * q1_pu, v1_sq - (r_pu * p1_pu + x_pu * q1_pu)))
    i_sq = (p1_pu ** 2 + q1_pu ** 2) / v1_sq
    p_loss_pu = float(i_sq * r_pu)
    q_loss_pu = float(i_sq * x_pu)
    return v1_pu, theta1_rad, p_loss_pu, q_loss_pu


def verify_two_bus_analytical() -> dict[str, float]:
    """Verify the Ybus Newton-Raphson solver against the closed-form 2-bus analytical solution."""
    from .network import BusSpec, BranchSpec, SubstationSpec

    r_pu = 0.088
    x_pu = 0.0926
    p1_mw = 6.2
    q1_mvar = -2.04
    v0_pu = 1.000

    v1_exact, th1_exact, ploss_exact_pu, qloss_exact_pu = solve_two_bus_analytical(
        v0_pu=v0_pu,
        r_pu=r_pu,
        x_pu=x_pu,
        p1_pu=p1_mw / BASE_MVA,
        q1_pu=q1_mvar / BASE_MVA,
    )

    net2 = NetworkModel(
        substation=SubstationSpec(p_fe_kw=0.0),
        buses=(
            BusSpec(0, "Slack", "T", 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_slack=True),
            BusSpec(1, "PCC", "T", 15.0, 14.2, 0.0, 0.0, 0.0, 0.0, is_pcc_candidate=True),
        ),
        branches=(
            BranchSpec(0, "Line", "T", 0, 1, "AFL-6", 14.2, r_pu, x_pu, 0.0, 20.0, 770.0, 1.0),
        ),
        feeders={},
        n_transformers_in_service=1,
        oltc_tap_ratio=1.0,
    )
    res = solve_ac_powerflow(
        net2,
        p_gen_mw=np.array([0.0, p1_mw]),
        q_gen_mvar=np.array([0.0, q1_mvar]),
        p_load_mw=np.zeros(2),
        q_load_mvar=np.zeros(2),
        v_slack_pu=v0_pu,
        regulate_gpz_15kv=False,
        tol=1e-13,
    )
    v_err = float(abs(res.v_mag_pu[1] - v1_exact))
    th_err = float(abs(res.v_ang_rad[1] - th1_exact))
    p_bal_err = float(abs(res.p_slack_mw + p1_mw - res.branch_p_loss_mw[0]))
    q_bal_err = float(abs(res.q_slack_mvar + q1_mvar - res.branch_q_loss_mvar[0]))
    ploss_err = float(abs(res.branch_p_loss_mw[0] - ploss_exact_pu * BASE_MVA))
    qloss_err = float(abs(res.branch_q_loss_mvar[0] - qloss_exact_pu * BASE_MVA))

    return {
        "r_pu": r_pu,
        "x_pu": x_pu,
        "p1_mw": p1_mw,
        "q1_mvar": q1_mvar,
        "v0_pu": v0_pu,
        "v1_exact_pu": v1_exact,
        "v1_nr_pu": float(res.v_mag_pu[1]),
        "v_abs_error_pu": v_err,
        "theta1_exact_rad": th1_exact,
        "theta1_nr_rad": float(res.v_ang_rad[1]),
        "theta_abs_error_rad": th_err,
        "p_balance_error_mw": p_bal_err,
        "q_balance_error_mvar": q_bal_err,
        "p_loss_exact_mw": ploss_exact_pu * BASE_MVA,
        "p_loss_nr_mw": float(res.branch_p_loss_mw[0]),
        "p_loss_error_mw": ploss_err,
        "q_loss_error_mvar": qloss_err,
        "nr_iterations": int(res.iterations),
    }
