"""Polish 110/15 kV GPZ substation and 15 kV radial feeder network models and Ybus builder."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BASE_MVA: float = 10.0
BASE_KV_MV: float = 15.0
BASE_KV_HV: float = 110.0
Z_BASE_MV_OHM: float = (BASE_KV_MV ** 2) / BASE_MVA  # 22.5 Ohm
Y_BASE_MV_S: float = 1.0 / Z_BASE_MV_OHM             # 0.044444 S
I_BASE_MV_KA: float = BASE_MVA / (np.sqrt(3.0) * BASE_KV_MV)  # 0.384900 kA = 384.9 A


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


LINE_TYPES: dict[str, LineTypeSpec] = {
    "AFL-6 120": LineTypeSpec(
        code="AFL-6 120",
        description="Overhead ACSR trunk exit AFL-6 120 mm2 (twin-circuit head section)",
        r_ohm_per_km=0.110,
        x_ohm_per_km=0.123,
        b_us_per_km=5.80,
        i_dd_a=520.0,
    ),
    "AFL-6 70": LineTypeSpec(
        code="AFL-6 70",
        description="Overhead ACSR rural trunk AFL-6 70 mm2",
        r_ohm_per_km=0.155,
        x_ohm_per_km=0.163,
        b_us_per_km=4.20,
        i_dd_a=480.0,
    ),
    "AFL-6 50": LineTypeSpec(
        code="AFL-6 50",
        description="Overhead ACSR rural spur AFL-6 50 mm2",
        r_ohm_per_km=0.178,
        x_ohm_per_km=0.183,
        b_us_per_km=3.80,
        i_dd_a=460.0,
    ),
    "NA2XS(F)2Y 3x150": LineTypeSpec(
        code="NA2XS(F)2Y 3x150",
        description="Underground Al XLPE cable NA2XS(F)2Y 3x150 mm2",
        r_ohm_per_km=0.146,
        x_ohm_per_km=0.115,
        b_us_per_km=76.0,
        i_dd_a=465.0,
    ),
    "NA2XS(F)2Y 3x240": LineTypeSpec(
        code="NA2XS(F)2Y 3x240",
        description="Underground Al XLPE suburban trunk NA2XS(F)2Y 3x240 mm2",
        r_ohm_per_km=0.072,
        x_ohm_per_km=0.189,
        b_us_per_km=98.0,
        i_dd_a=518.0,
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
    limiting_mechanism: str


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
    s_max_override_mva: float | None = None,
) -> BranchSpec:
    lt = LINE_TYPES[line_type_key]
    r_ohm = lt.r_ohm_per_km * length_km
    x_ohm = lt.x_ohm_per_km * length_km
    b_s = (lt.b_us_per_km * 1e-6) * length_km
    r_pu = r_ohm / Z_BASE_MV_OHM
    x_pu = x_ohm / Z_BASE_MV_OHM
    b_pu = b_s / Y_BASE_MV_S
    s_max = s_max_override_mva if s_max_override_mva is not None else lt.s_thermal_mva
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
      - Buses 2..6: Feeder F1 (Rural overhead AFL-6 70 + AFL-6 50, 14.2 km, R/X ~ 0.95)
      - Buses 7..11: Feeder F2 (Mixed cable NA2XS(F)2Y 3x150 + overhead AFL-6 70, 9.6 km)
      - Buses 12..16: Feeder F3 (Suburban underground cable NA2XS(F)2Y 3x240, 6.4 km, R/X ~ 0.38)
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

    # Feeder F1: Rural long overhead (14.2 km total, R/X ~ 0.95, voltage-rise limited)
    f1_segs = [
        ("F1_N1", 2.8, "AFL-6 120", 0.25, 0.09, 0.08, 0.03, 18.5),
        ("F1_N2", 3.0, "AFL-6 70", 0.28, 0.10, 0.09, 0.03, 18.0),
        ("F1_N3", 3.0, "AFL-6 70", 0.30, 0.11, 0.10, 0.03, 17.5),
        ("F1_N4", 2.8, "AFL-6 50", 0.26, 0.09, 0.09, 0.03, 17.2),
        ("F1_PCC", 2.6, "AFL-6 50", 0.26, 0.09, 0.09, 0.03, 17.0),
    ]
    dist = 0.0
    prev_bus = 1
    f1_bus_ids = []
    f1_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn, s_lim) in enumerate(f1_segs):
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
            _make_line_branch(brid, f"F1_L{i+1}", "F1", prev_bus, bid, ltype, seg_km, s_max_override_mva=s_lim)
        )
        f1_bus_ids.append(bid)
        f1_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F2: Mixed underground cable NA2XS(F)2Y 3x150 + overhead AFL-6 70 (9.6 km total)
    f2_segs = [
        ("F2_N1", 2.2, "NA2XS(F)2Y 3x150", 0.55, 0.18, 0.20, 0.06, 12.1),
        ("F2_N2", 2.4, "NA2XS(F)2Y 3x150", 0.50, 0.17, 0.18, 0.06, 12.1),
        ("F2_N3", 1.8, "AFL-6 70", 0.45, 0.15, 0.16, 0.05, 12.0),
        ("F2_N4", 1.6, "AFL-6 70", 0.45, 0.15, 0.16, 0.05, 12.0),
        ("F2_PCC", 1.6, "AFL-6 70", 0.45, 0.15, 0.15, 0.05, 11.9),
    ]
    dist = 0.0
    prev_bus = 1
    f2_bus_ids = []
    f2_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn, s_lim) in enumerate(f2_segs):
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
            _make_line_branch(brid, f"F2_L{i+1}", "F2", prev_bus, bid, ltype, seg_km, s_max_override_mva=s_lim)
        )
        f2_bus_ids.append(bid)
        f2_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F3: Suburban underground cable NA2XS(F)2Y 3x240 (6.4 km total, R/X ~ 0.38, thermally limited)
    f3_segs = [
        ("F3_N1", 1.4, "NA2XS(F)2Y 3x240", 0.70, 0.23, 0.25, 0.08, 13.55),
        ("F3_N2", 1.4, "NA2XS(F)2Y 3x240", 0.70, 0.23, 0.25, 0.08, 13.55),
        ("F3_N3", 1.2, "NA2XS(F)2Y 3x240", 0.65, 0.21, 0.24, 0.08, 13.55),
        ("F3_N4", 1.2, "NA2XS(F)2Y 3x240", 0.65, 0.21, 0.23, 0.07, 13.55),
        ("F3_PCC", 1.2, "NA2XS(F)2Y 3x240", 0.60, 0.20, 0.23, 0.07, 13.55),
    ]
    dist = 0.0
    prev_bus = 1
    f3_bus_ids = []
    f3_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn, s_lim) in enumerate(f3_segs):
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
            _make_line_branch(brid, f"F3_L{i+1}", "F3", prev_bus, bid, ltype, seg_km, s_max_override_mva=s_lim)
        )
        f3_bus_ids.append(bid)
        f3_branch_ids.append(brid)
        prev_bus = bid

    # Feeder F4: Industrial/municipal background load feeder (5.1 km total)
    f4_segs = [
        ("F4_N1", 1.8, "NA2XS(F)2Y 3x240", 1.60, 0.62, 0.75, 0.28, 16.0),
        ("F4_N2", 1.7, "NA2XS(F)2Y 3x240", 1.60, 0.62, 0.75, 0.28, 16.0),
        ("F4_N3", 1.6, "NA2XS(F)2Y 3x240", 1.60, 0.62, 0.70, 0.26, 16.0),
    ]
    dist = 0.0
    prev_bus = 1
    f4_bus_ids = []
    f4_branch_ids = []
    for i, (bname, seg_km, ltype, p_pk, q_pk, p_mn, q_mn, s_lim) in enumerate(f4_segs):
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
            _make_line_branch(brid, f"F4_L{i+1}", "F4", prev_bus, bid, ltype, seg_km, s_max_override_mva=s_lim)
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
            description="AFL-6 70 mm2 + AFL-6 50 mm2 overhead rural line, 14.2 km, high R/X ~ 0.95",
            total_length_km=14.2,
            rx_ratio=_feeder_rx(f1_branch_ids),
            pcc_bus_id=f1_bus_ids[-1],
            bus_ids=tuple(f1_bus_ids),
            branch_ids=tuple(f1_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f1_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f1_bus_ids), 2),
            limiting_mechanism="PCC voltage rise (+5.0 %)",
        ),
        "F2": FeederSpec(
            code="F2",
            name="Feeder F2 (Mixed Cable/Overhead 9.6 km)",
            description="NA2XS(F)2Y 3x150 mm2 underground exit + AFL-6 70 mm2 overhead, 9.6 km",
            total_length_km=9.6,
            rx_ratio=_feeder_rx(f2_branch_ids),
            pcc_bus_id=f2_bus_ids[-1],
            bus_ids=tuple(f2_bus_ids),
            branch_ids=tuple(f2_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f2_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f2_bus_ids), 2),
            limiting_mechanism="Voltage rise -> Cable thermal boundary",
        ),
        "F3": FeederSpec(
            code="F3",
            name="Feeder F3 (Suburban Cable 6.4 km)",
            description="NA2XS(F)2Y 3x240 mm2 underground XLPE cable, 6.4 km, low R/X ~ 0.38",
            total_length_km=6.4,
            rx_ratio=_feeder_rx(f3_branch_ids),
            pcc_bus_id=f3_bus_ids[-1],
            bus_ids=tuple(f3_bus_ids),
            branch_ids=tuple(f3_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f3_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f3_bus_ids), 2),
            limiting_mechanism="Branch thermal ampacity (I <= I_dd)",
        ),
        "F4": FeederSpec(
            code="F4",
            name="Feeder F4 (Industrial Load 5.1 km)",
            description="NA2XS(F)2Y 3x240 mm2 municipal/industrial background load feeder, 5.1 km",
            total_length_km=5.1,
            rx_ratio=_feeder_rx(f4_branch_ids),
            pcc_bus_id=f4_bus_ids[-1],
            bus_ids=tuple(f4_bus_ids),
            branch_ids=tuple(f4_branch_ids),
            p_load_min_mw=round(sum(buses[b].p_load_min_mw for b in f4_bus_ids), 2),
            p_load_peak_mw=round(sum(buses[b].p_load_peak_mw for b in f4_bus_ids), 2),
            limiting_mechanism="Background load sink",
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
