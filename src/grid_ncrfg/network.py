"""Polish 110/15 kV GPZ substation and 15 kV radial feeder network models and Ybus builder.

Conductor data sources (catalogue values, to be verified against the supplier of an actual project):
  - AFL-6 bare ACSR (PN-EN 50182): DC resistance at 20 C and overall diameter from the Tele-Fonika Kable
    "AFL" catalogue sheet and the Eltrim Kable AFL-6 data sheet (max continuous conductor temperature 80 C).
    Neither sheet gives reactance or ampacity, so series reactance and shunt susceptance are computed from the
    conductor radius and an assumed geometric mean phase spacing, and ampacity from the IEEE Std 738-2012
    steady-state heat balance under the summer weather assumptions in `acsr_ampacity_a`.
  - NA2XS(F)2Y 12/20 kV single-core Al XLPE cables (three cores in trefoil): DC resistance at 20 C, capacitance,
    trefoil inductance and continuous current rating in ground (trefoil, 0.7 m depth, soil 1.0 K m/W, screens
    bonded at both ends) from the NKT data sheet "NA2XS(F)2Y (XUHAKXS) 12/20 kV" (catalogue pp. 207-211).
Resistances are kept at 20 C (no temperature correction), which slightly understates voltage rise and losses.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BASE_MVA: float = 10.0
BASE_KV_MV: float = 15.0
BASE_KV_HV: float = 110.0
Z_BASE_MV_OHM: float = (BASE_KV_MV ** 2) / BASE_MVA  # 22.5 Ohm
Y_BASE_MV_S: float = 1.0 / Z_BASE_MV_OHM             # 0.044444 S
I_BASE_MV_KA: float = BASE_MVA / (np.sqrt(3.0) * BASE_KV_MV)  # 0.384900 kA = 384.9 A

F_HZ: float = 50.0
MU0_OVER_2PI: float = 2.0e-7            # H/m
EPS0: float = 8.854187817e-12           # F/m
OVERHEAD_PHASE_SPACING_M: float = 1.0   # assumed geometric mean phase spacing of a 15 kV overhead line


@dataclass(frozen=True)
class LineTypeSpec:
    """Standard Polish 15 kV overhead conductor (AFL-6) or underground XLPE cable (NA2XS(F)2Y)."""

    code: str
    description: str
    r_ohm_per_km: float
    x_ohm_per_km: float
    b_us_per_km: float
    i_dd_a: float  # Continuous thermal ampacity [A]

    @property
    def rx_ratio(self) -> float:
        return self.r_ohm_per_km / self.x_ohm_per_km

    @property
    def s_thermal_mva(self) -> float:
        return float(np.sqrt(3.0) * BASE_KV_MV * (self.i_dd_a / 1000.0))


def acsr_ampacity_a(
    d_mm: float,
    r20_ohm_per_km: float,
    t_amb_c: float = 35.0,
    t_max_c: float = 80.0,
    wind_m_s: float = 0.6,
    solar_w_m2: float = 1000.0,
    absorptivity: float = 0.5,
    emissivity: float = 0.5,
    alpha_r_per_k: float = 0.00403,
) -> float:
    """Steady-state ampacity [A] of a bare overhead conductor from the IEEE Std 738-2012 heat balance.

    I = sqrt((q_c + q_r - q_s) / R(T_max)) with forced/natural convection q_c (wind perpendicular to the
    conductor, sea level), radiation q_r and solar gain q_s, all in W/m. Defaults are a hot, calm summer
    afternoon (35 C, 0.6 m/s, 1000 W/m2) at the 80 C continuous limit given for AFL-6 by its manufacturer.
    """
    d_m = d_mm / 1000.0
    dt = t_max_c - t_amb_c
    t_film = 0.5 * (t_max_c + t_amb_c)
    mu_f = 1.458e-6 * (t_film + 273.0) ** 1.5 / (t_film + 383.4)
    rho_f = 1.293 / (1.0 + 0.00367 * t_film)
    k_f = 2.424e-2 + 7.477e-5 * t_film - 4.407e-9 * t_film ** 2
    n_re = d_m * rho_f * wind_m_s / mu_f
    q_c = max(
        (1.01 + 1.35 * n_re ** 0.52) * k_f * dt,
        0.754 * n_re ** 0.6 * k_f * dt,
        3.645 * rho_f ** 0.5 * d_m ** 0.75 * dt ** 1.25,
    )
    q_r = 17.8 * d_m * emissivity * (((t_max_c + 273.0) / 100.0) ** 4 - ((t_amb_c + 273.0) / 100.0) ** 4)
    q_s = absorptivity * solar_w_m2 * d_m
    r_ohm_per_m = r20_ohm_per_km / 1000.0 * (1.0 + alpha_r_per_k * (t_max_c - 20.0))
    return float(np.sqrt((q_c + q_r - q_s) / r_ohm_per_m))


def _overhead_acsr(code: str, description: str, d_mm: float, r20_ohm_per_km: float) -> LineTypeSpec:
    r_m = d_mm / 2000.0
    gmr_m = np.exp(-0.25) * r_m
    x_ohm_per_km = 2.0 * np.pi * F_HZ * MU0_OVER_2PI * np.log(OVERHEAD_PHASE_SPACING_M / gmr_m) * 1000.0
    c_f_per_km = 2.0 * np.pi * EPS0 / np.log(OVERHEAD_PHASE_SPACING_M / r_m) * 1000.0
    return LineTypeSpec(
        code=code,
        description=description,
        r_ohm_per_km=r20_ohm_per_km,
        x_ohm_per_km=float(x_ohm_per_km),
        b_us_per_km=float(2.0 * np.pi * F_HZ * c_f_per_km * 1e6),
        i_dd_a=acsr_ampacity_a(d_mm, r20_ohm_per_km),
    )


def _xlpe_cable(code: str, description: str, r20_ohm_per_km: float, l_mh_per_km: float,
                c_uf_per_km: float, i_ground_a: float) -> LineTypeSpec:
    return LineTypeSpec(
        code=code,
        description=description,
        r_ohm_per_km=r20_ohm_per_km,
        x_ohm_per_km=float(2.0 * np.pi * F_HZ * l_mh_per_km * 1e-3),
        b_us_per_km=float(2.0 * np.pi * F_HZ * c_uf_per_km),
        i_dd_a=i_ground_a,
    )


LINE_TYPES: dict[str, LineTypeSpec] = {
    "AFL-6 120": _overhead_acsr("AFL-6 120", "Overhead ACSR AFL-6 120 mm2 (123-AL1/21-ST1A), trunk exit", 15.7, 0.2356),
    "AFL-6 70": _overhead_acsr("AFL-6 70", "Overhead ACSR AFL-6 70 mm2 (66-AL1/11-ST1A), rural trunk", 11.3, 0.4324),
    "AFL-6 50": _overhead_acsr("AFL-6 50", "Overhead ACSR AFL-6 50 mm2 (48-AL1/8-ST1A), rural spur", 9.6, 0.5939),
    "NA2XS(F)2Y 1x150/25": _xlpe_cable(
        "NA2XS(F)2Y 1x150/25", "Underground Al XLPE cable 12/20 kV, 3 x 1x150/25 in trefoil", 0.2060, 0.37, 0.25, 320.0
    ),
    "NA2XS(F)2Y 1x240/25": _xlpe_cable(
        "NA2XS(F)2Y 1x240/25", "Underground Al XLPE cable 12/20 kV, 3 x 1x240/25 in trefoil", 0.1250, 0.34, 0.29, 420.0
    ),
}


@dataclass(frozen=True)
class SubstationSpec:
    """Polish GPZ 110/15 kV substation with two parallel 25 MVA OLTC transformers."""

    name: str = "GPZ 110/15 kV Central Poland"
    n_transformers: int = 2
    s_nom_each_mva: float = 25.0
    u_hv_kv: float = 110.0
    u_mv_kv: float = 15.75
    uk_pct: float = 11.5
    p_cu_kw: float = 125.0
    p_fe_kw: float = 22.0
    oltc_steps: int = 9
    oltc_step_pct: float = 1.78
    v_target_mv_pu: float = 1.000

    @property
    def s_n0_mva(self) -> float:
        return self.n_transformers * self.s_nom_each_mva

    @property
    def s_n1_mva(self) -> float:
        return self.s_nom_each_mva

    def transformer_z_pu(self, n_in_service: int = 2) -> complex:
        """Return transformer series impedance on BASE_MVA = 10 MVA."""
        ur_pct = 100.0 * (self.p_cu_kw / 1000.0) / self.s_nom_each_mva
        ux_pct = float(np.sqrt(max(0.0, self.uk_pct ** 2 - ur_pct ** 2)))
        r_1 = (ur_pct / 100.0) * (BASE_MVA / self.s_nom_each_mva)
        x_1 = (ux_pct / 100.0) * (BASE_MVA / self.s_nom_each_mva)
        return complex(r_1, x_1) / max(1, int(n_in_service))


@dataclass(frozen=True)
class BusSpec:
    """Network bus specification."""

    bus_id: int
    name: str
    feeder: str
    vn_kv: float
    distance_km: float
    p_load_peak_mw: float
    q_load_peak_mvar: float
    p_load_min_mw: float
    q_load_min_mvar: float
    is_slack: bool = False
    is_pcc_candidate: bool = False


@dataclass(frozen=True)
class BranchSpec:
    """Network branch (transformer or 15 kV line segment) in per-unit on BASE_MVA."""

    branch_id: int
    name: str
    feeder: str
    from_bus: int
    to_bus: int
    line_type: str
    length_km: float
    r_pu: float
    x_pu: float
    b_pu: float
    s_max_mva: float
    i_max_a: float
    tap_ratio: float = 1.0


@dataclass(frozen=True)
class FeederSpec:
    """Summary metadata of each 15 kV radial feeder."""

    code: str
    name: str
    description: str
    total_length_km: float
    rx_ratio: float
    pcc_bus_id: int
    bus_ids: tuple[int, ...]
    branch_ids: tuple[int, ...]
    p_load_min_mw: float
    p_load_peak_mw: float


@dataclass(frozen=True)
class NetworkModel:
    """Complete Polish GPZ 110/15 kV + 4 radial 15 kV feeders model."""

    substation: SubstationSpec
    buses: tuple[BusSpec, ...]
    branches: tuple[BranchSpec, ...]
    feeders: dict[str, FeederSpec]
    n_transformers_in_service: int = 2
    oltc_tap_ratio: float = 1.0


def _make_line_branch(
    branch_id: int,
    name: str,
    feeder: str,
    from_bus: int,
    to_bus: int,
    line_type_key: str,
    length_km: float,
) -> BranchSpec:
    lt = LINE_TYPES[line_type_key]
    r_ohm = lt.r_ohm_per_km * length_km
    x_ohm = lt.x_ohm_per_km * length_km
    b_s = (lt.b_us_per_km * 1e-6) * length_km
    r_pu = r_ohm / Z_BASE_MV_OHM
    x_pu = x_ohm / Z_BASE_MV_OHM
    b_pu = b_s / Y_BASE_MV_S
    s_max = lt.s_thermal_mva
    i_max = float(s_max * 1000.0 / (np.sqrt(3.0) * BASE_KV_MV))
    return BranchSpec(
        branch_id=branch_id,
        name=name,
        feeder=feeder,
        from_bus=from_bus,
        to_bus=to_bus,
        line_type=lt.code,
        length_km=length_km,
        r_pu=r_pu,
        x_pu=x_pu,
        b_pu=b_pu,
        s_max_mva=s_max,
        i_max_a=i_max,
        tap_ratio=1.0,
    )


def build_polish_gpz_network(
    n_transformers: int = 2,
    oltc_tap_step: int = 0,
) -> NetworkModel:
    """Construct the Polish 110/15 kV GPZ substation and four 15 kV feeders (F1..F4).

    Topology:
      - Bus 0: GPZ_110kV (Slack bus, 110 kV)
      - Bus 1: GPZ_15kV (15 kV two-section busbar fed via TR1 + TR2)
      - Buses 2..6: Feeder F1 (Rural overhead AFL-6 120 / 70 / 50, 14.2 km)
      - Buses 7..11: Feeder F2 (Mixed cable NA2XS(F)2Y 1x150/25 + overhead AFL-6 70, 9.6 km)
      - Buses 12..16: Feeder F3 (Suburban underground cable NA2XS(F)2Y 1x240/25, 6.4 km)
      - Buses 17..19: Feeder F4 (Industrial/municipal background load feeder, 5.1 km)
    """
    sub = SubstationSpec()
    tap_ratio = 1.0 + float(np.clip(oltc_tap_step, -sub.oltc_steps, sub.oltc_steps)) * (sub.oltc_step_pct / 100.0)
    z_tr = sub.transformer_z_pu(n_in_service=n_transformers)
    s_tr_max = sub.s_n0_mva if n_transformers >= 2 else sub.s_n1_mva

    buses: list[BusSpec] = [
        BusSpec(
            bus_id=0,
            name="GPZ_110kV",
            feeder="GPZ",
            vn_kv=110.0,
            distance_km=0.0,
            p_load_peak_mw=0.0,
            q_load_peak_mvar=0.0,
            p_load_min_mw=0.0,
            q_load_min_mvar=0.0,
            is_slack=True,
        ),
        BusSpec(
            bus_id=1,
            name="GPZ_15kV",
            feeder="GPZ",
            vn_kv=15.0,
            distance_km=0.0,
            p_load_peak_mw=0.12,
            q_load_peak_mvar=0.04,
            p_load_min_mw=0.05,
            q_load_min_mvar=0.02,
            is_slack=False,
        ),
    ]

    branches: list[BranchSpec] = [
        BranchSpec(
            branch_id=0,
            name="TR_110_15kV",
            feeder="GPZ",
            from_bus=0,
            to_bus=1,
            line_type=f"{n_transformers}x 25 MVA 110/15.75 kV",
            length_km=0.0,
            r_pu=float(z_tr.real),
            x_pu=float(z_tr.imag),
            b_pu=0.0,
            s_max_mva=s_tr_max,
            i_max_a=float(s_tr_max * 1000.0 / (np.sqrt(3.0) * BASE_KV_MV)),
            tap_ratio=tap_ratio,
        )
    ]

    # Feeder F1: Rural long overhead line (14.2 km total)
    f1_segs = [
        ("F1_N1", 2.8, "AFL-6 120", 0.25, 0.09, 0.08, 0.03),
        ("F1_N2", 3.0, "AFL-6 70", 0.28, 0.10, 0.09, 0.03),
        ("F1_N3", 3.0, "AFL-6 70", 0.30, 0.11, 0.10, 0.03),
        ("F1_N4", 2.8, "AFL-6 50", 0.26, 0.09, 0.09, 0.03),
        ("F1_PCC", 2.6, "AFL-6 50", 0.26, 0.09, 0.09, 0.03),
    ]
    dist = 0.0
    prev_bus = 1
    f1_bus_ids = []
    f1_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn) in enumerate(f1_segs):
        dist = round(dist + seg_km, 2)
        bid = len(buses)
        is_pcc = i == len(f1_segs) - 1
        buses.append(
            BusSpec(
                bus_id=bid,
                name=bname,
                feeder="F1",
                vn_kv=15.0,
                distance_km=dist,
                p_load_peak_mw=p_pk,
                q_load_peak_mvar=q_pk,
                p_load_min_mw=p_mn,
                q_load_min_mvar=q_mn,
                is_pcc_candidate=is_pcc,
            )
        )
        brid = len(branches)
        branches.append(
            _make_line_branch(brid, f"F1_L{i+1}", "F1", prev_bus, bid, ltype, seg_km)
        )
        f1_bus_ids.append(bid)
        f1_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F2: Mixed underground cable NA2XS(F)2Y 1x150/25 + overhead AFL-6 70 (9.6 km total)
    f2_segs = [
        ("F2_N1", 2.2, "NA2XS(F)2Y 1x150/25", 0.55, 0.18, 0.20, 0.06),
        ("F2_N2", 2.4, "NA2XS(F)2Y 1x150/25", 0.50, 0.17, 0.18, 0.06),
        ("F2_N3", 1.8, "AFL-6 70", 0.45, 0.15, 0.16, 0.05),
        ("F2_N4", 1.6, "AFL-6 70", 0.45, 0.15, 0.16, 0.05),
        ("F2_PCC", 1.6, "AFL-6 70", 0.45, 0.15, 0.15, 0.05),
    ]
    dist = 0.0
    prev_bus = 1
    f2_bus_ids = []
    f2_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn) in enumerate(f2_segs):
        dist = round(dist + seg_km, 2)
        bid = len(buses)
        is_pcc = i == len(f2_segs) - 1
        buses.append(
            BusSpec(
                bus_id=bid,
                name=bname,
                feeder="F2",
                vn_kv=15.0,
                distance_km=dist,
                p_load_peak_mw=p_pk,
                q_load_peak_mvar=q_pk,
                p_load_min_mw=p_mn,
                q_load_min_mvar=q_mn,
                is_pcc_candidate=is_pcc,
            )
        )
        brid = len(branches)
        branches.append(
            _make_line_branch(brid, f"F2_L{i+1}", "F2", prev_bus, bid, ltype, seg_km)
        )
        f2_bus_ids.append(bid)
        f2_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F3: Suburban underground cable NA2XS(F)2Y 1x240/25 (6.4 km total)
    f3_segs = [
        ("F3_N1", 1.4, "NA2XS(F)2Y 1x240/25", 0.70, 0.23, 0.25, 0.08),
        ("F3_N2", 1.4, "NA2XS(F)2Y 1x240/25", 0.70, 0.23, 0.25, 0.08),
        ("F3_N3", 1.2, "NA2XS(F)2Y 1x240/25", 0.65, 0.21, 0.24, 0.08),
        ("F3_N4", 1.2, "NA2XS(F)2Y 1x240/25", 0.65, 0.21, 0.23, 0.07),
        ("F3_PCC", 1.2, "NA2XS(F)2Y 1x240/25", 0.60, 0.20, 0.23, 0.07),
    ]
    dist = 0.0
    prev_bus = 1
    f3_bus_ids = []
    f3_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn) in enumerate(f3_segs):
        dist = round(dist + seg_km, 2)
        bid = len(buses)
        is_pcc = i == len(f3_segs) - 1
        buses.append(
            BusSpec(
                bus_id=bid,
                name=bname,
                feeder="F3",
                vn_kv=15.0,
                distance_km=dist,
                p_load_peak_mw=p_pk,
                q_load_peak_mvar=q_pk,
                p_load_min_mw=p_mn,
                q_load_min_mvar=q_mn,
                is_pcc_candidate=is_pcc,
            )
        )
        brid = len(branches)
        branches.append(
            _make_line_branch(brid, f"F3_L{i+1}", "F3", prev_bus, bid, ltype, seg_km)
        )
        f3_bus_ids.append(bid)
        f3_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F4: Industrial/municipal background load feeder (5.1 km total)
    f4_segs = [
        ("F4_N1", 1.8, "NA2XS(F)2Y 1x240/25", 1.60, 0.62, 0.75, 0.28),
        ("F4_N2", 1.7, "NA2XS(F)2Y 1x240/25", 1.60, 0.62, 0.75, 0.28),
        ("F4_N3", 1.6, "NA2XS(F)2Y 1x240/25", 1.60, 0.62, 0.70, 0.26),
    ]
    dist = 0.0
    prev_bus = 1
    f4_bus_ids = []
    f4_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn) in enumerate(f4_segs):
        dist = round(dist + seg_km, 2)
        bid = len(buses)
        buses.append(
            BusSpec(
                bus_id=bid,
                name=bname,
                feeder="F4",
                vn_kv=15.0,
                distance_km=dist,
                p_load_peak_mw=p_pk,
                q_load_peak_mvar=q_pk,
                p_load_min_mw=p_mn,
                q_load_min_mvar=q_mn,
                is_pcc_candidate=False,
            )
        )
        brid = len(branches)
        branches.append(
            _make_line_branch(brid, f"F4_L{i+1}", "F4", prev_bus, bid, ltype, seg_km)
        )
        f4_bus_ids.append(bid)
        f4_branch_ids.append(brid)
        prev_bus = bid

    def _feeder_rx(br_ids: list[int]) -> float:
        r_sum = sum(branches[k].r_pu for k in br_ids)
        x_sum = sum(branches[k].x_pu for k in br_ids)
        return round(r_sum / x_sum, 3)

    feeders = {
        "F1": FeederSpec(
            code="F1",
            name="Feeder F1 (Rural Overhead 14.2 km)",
            description="AFL-6 120 / 70 / 50 mm2 overhead rural line, 14.2 km",
            total_length_km=round(sum(seg[1] for seg in f1_segs), 2),
            rx_ratio=_feeder_rx(f1_branch_ids),
            pcc_bus_id=f1_bus_ids[-1],
            bus_ids=tuple(f1_bus_ids),
            branch_ids=tuple(f1_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f1_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f1_bus_ids), 2),
        ),
        "F2": FeederSpec(
            code="F2",
            name="Feeder F2 (Mixed Cable/Overhead 9.6 km)",
            description="NA2XS(F)2Y 1x150/25 underground exit + AFL-6 70 mm2 overhead, 9.6 km",
            total_length_km=round(sum(seg[1] for seg in f2_segs), 2),
            rx_ratio=_feeder_rx(f2_branch_ids),
            pcc_bus_id=f2_bus_ids[-1],
            bus_ids=tuple(f2_bus_ids),
            branch_ids=tuple(f2_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f2_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f2_bus_ids), 2),
        ),
        "F3": FeederSpec(
            code="F3",
            name="Feeder F3 (Suburban Cable 6.4 km)",
            description="NA2XS(F)2Y 1x240/25 underground XLPE cable, 6.4 km",
            total_length_km=round(sum(seg[1] for seg in f3_segs), 2),
            rx_ratio=_feeder_rx(f3_branch_ids),
            pcc_bus_id=f3_bus_ids[-1],
            bus_ids=tuple(f3_bus_ids),
            branch_ids=tuple(f3_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f3_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f3_bus_ids), 2),
        ),
        "F4": FeederSpec(
            code="F4",
            name="Feeder F4 (Industrial Load 5.1 km)",
            description="NA2XS(F)2Y 1x240/25 municipal/industrial background load feeder, 5.1 km",
            total_length_km=round(sum(seg[1] for seg in f4_segs), 2),
            rx_ratio=_feeder_rx(f4_branch_ids),
            pcc_bus_id=f4_bus_ids[-1],
            bus_ids=tuple(f4_bus_ids),
            branch_ids=tuple(f4_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f4_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f4_bus_ids), 2),
        ),
    }

    return NetworkModel(
        substation=sub,
        buses=tuple(buses),
        branches=tuple(branches),
        feeders=feeders,
        n_transformers_in_service=n_transformers,
        oltc_tap_ratio=tap_ratio,
    )


def build_Ybus(net: NetworkModel, tap_override: float | None = None) -> np.ndarray:
    """Assemble complex nodal admittance matrix Ybus (n_bus x n_bus) in per-unit on BASE_MVA.

    Uses the standard pi-equivalent model for lines and off-nominal tap ratio t:1 on the
    primary (from_bus) side of the 110/15 kV transformer.
    """
    n = len(net.buses)
    Ybus = np.zeros((n, n), dtype=complex)
    for br in net.branches:
        i = br.from_bus
        j = br.to_bus
        z = complex(br.r_pu, br.x_pu)
        y_ser = 1.0 / z
        y_sh = 1j * (br.b_pu / 2.0)
        t = tap_override if (br.branch_id == 0 and tap_override is not None) else br.tap_ratio

        # Standard off-nominal transformer / pi-line element
        Ybus[i, i] += (y_ser + y_sh) / (t ** 2)
        Ybus[j, j] += y_ser + y_sh
        Ybus[i, j] -= y_ser / t
        Ybus[j, i] -= y_ser / t

    return Ybus
