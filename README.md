# Grid connection studies and non-firm capacity on Polish MV/110 kV feeders

[![tests](https://github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg/actions/workflows/ci.yml/badge.svg)](https://github.com/kwwaleszczynski-del/pl-grid-connection-ncrfg/actions/workflows/ci.yml)

Refusals of grid connection conditions (*odmowy wydania warunków przyłączenia*) for medium-voltage (15 kV) solar PV
and onshore wind plants in Poland usually come down to two physical limits of the distribution network: voltage rise
at the point of common coupling (PCC) on long overhead rural feeders, and reverse power flow through the 110/15 kV
substation transformers when one of them is out of service (`N-1`).

This repository models a **synthetic** Polish 110/15 kV primary substation (*Główny Punkt Zasilający*, GPZ, two
parallel 25 MVA on-load tap-changing transformers) and four radial 15 kV feeders built from catalogue conductor data,
and solves it with a polar Newton-Raphson AC power flow (`Ybus`). On top of the solver it evaluates:

1. **Hosting capacity by feeder type and reactive control**: largest plant that keeps every bus at or below
   $`1.05\text{ pu}`$ and every branch within its ampacity, by binary search over the AC power flow, for unity power
   factor, a $`\cos\varphi(P)`$ characteristic and a $`Q(U)`$ droop (control-mode families of NC RfG and EN 50549-2).
2. **Annual reactive energy and losses** of $`\cos\varphi(P)`$ vs $`Q(U)`$ from an hourly power flow over the year.
3. **Export caps**: the annual energy cost of capping active export below installed capacity, with and without a
   co-located battery (hourly state-of-charge model).
4. **GPZ under `N-1` over a year**: hourly power flow with one transformer out for a firm PV plant, an oversized PV
   plant behind an export cap, and PV + wind + battery sharing one connection (cable pooling).
5. **Solver verification** against the closed-form 2-bus power flow solution.

Every number below is produced by `python -m grid_ncrfg run` and stored in [`results/`](results/); the network, the
load and the wind profile are synthetic and PV is one PVGIS year, so the results illustrate mechanisms, not a real
substation or a DSO decision.

Interactive dashboard: https://kwwaleszczynski-del.github.io/pl-grid-connection-ncrfg/

## Key results

### 1. Hosting capacity by feeder and reactive control regime

Minimum daytime load, plant at rated output, GPZ 15 kV busbar held at $`1.000\text{ pu}`$ by the OLTC, one 25 MVA
transformer out of service (`N-1`). Feeder plants sit at the feeder end (75 % at the last bus, 25 % one bus upstream);
the GPZ row is new generation on the 15 kV busbar through a dedicated bay, with 8 MW of
existing busbar PV at full output. Capacities are rounded down to 10 kW.

| Location | Topology | 1. $`\cos\varphi = 1`$ | 2. $`\cos\varphi(P)`$, 0.95 ind at $`P_n`$ | 3. $`Q(U)`$ droop | Binding constraint |
|---|---|---:|---:|---:|---|
| **Feeder F1** | 14.2 km overhead `AFL-6 120/70/50`, $`R/X = 1.33`$ | 2.30 MW | 3.21 MW (+39.6 %) | 3.11 MW (+35.2 %) | PCC voltage (+5.0 %) |
| **Feeder F2** | 9.6 km `NA2XS(F)2Y 1x150` + `AFL-6 70`, $`R/X = 1.39`$ | 4.64 MW | 6.31 MW (+36.0 %) | 6.18 MW (+33.2 %) | PCC voltage (+5.0 %) |
| **Feeder F3** | 6.4 km cable `NA2XS(F)2Y 1x240`, $`R/X = 1.17`$ | 11.41 MW | 10.77 MW (-5.6 %) | 11.33 MW (-0.7 %) | Thermal ampacity (I_dd) |
| **GPZ 15 kV busbar** | `N-1`: 1 x 25 MVA in service, 8 MW existing PV | 21.54 MW | 19.59 MW (-9.1 %) | 21.54 MW (+0.0 %) | N-1 transformer (25 MVA) |

- **Reactive control lifts the long overhead feeder by 39.6 % ($`\cos\varphi(P)`$) and
  35.2 % ($`Q(U)`$), from 2.30 MW to 3.21 MW and
  3.11 MW.** F1 ($`R = 6.46\ \Omega`$, $`X = 4.84\ \Omega`$) is
  voltage-limited; absorbing reactive power through $`X`$ offsets part of the $`P R`$ voltage rise. At the hosting
  capacity both modes absorb close to $`0.33\,P_n`$, so they end up within 100 kW of
  each other; $`\cos\varphi(P)`$ is slightly ahead because its full absorption applies at both plant buses, while
  $`Q(U)`$ at the upstream bus responds to a lower local voltage.
- **On the cable feeder reactive absorption costs capacity.** F3 is thermally limited (PCC voltage only
  1.0340 pu at full ampacity). $`\cos\varphi(P)`$ adds 3.54 Mvar of
  reactive current and lowers the capacity by 5.6 %, while $`Q(U)`$ absorbs
  1.14 Mvar and costs 0.7 %.
- **Modern Al XLPE cables do not have a low $`R/X`$.** With catalogue data the F3 cable has $`R/X = 1.17`$;
  it hosts more than F1 because its impedance is 8.1 times smaller in $`R`$, not
  because of the ratio.
- **At the regulated GPZ busbar $`Q(U)`$ stays idle** (the OLTC holds the voltage inside the deadband), so it matches
  unity power factor at 21.54 MW. $`\cos\varphi(P)`$ absorbs 6.44 Mvar
  that the remaining 25 MVA transformer must also carry, which lowers the busbar capacity by 9.1 %.

![Hosting capacity and F1 voltage profile](docs/img/hosting_capacity.png)

### 2. Annual reactive energy and losses: $`Q(U)`$ deadband vs $`\cos\varphi(P)`$

Hourly AC power flow over all 4,082 generating hours of the PVGIS year for a
2.30 MW PV plant on Feeder F1 (the unity-power-factor hosting capacity, so all three regimes are
feasible), both GPZ transformers in service, load following the hourly load profile:

| Control regime | Hours absorbing Q | Reactive energy absorbed | Extra feeder $`I^2 R`$ losses vs $`\cos\varphi = 1`$ | Highest PCC voltage |
|---|---:|---:|---:|---:|
| $`\cos\varphi = 1`$ | 0 h | 0.0 Mvarh | - | 1.0440 pu |
| $`\cos\varphi(P)`$ | 1,103 h | 408.4 Mvarh | 9.222 MWh/yr | 1.0298 pu |
| $`Q(U)`$, deadband $`[0.98, 1.02]\text{ pu}`$ | 785 h | 118.4 Mvarh | 1.798 MWh/yr | 1.0365 pu |

- $`Q(U)`$ absorbs 71.0 % less reactive energy than $`\cos\varphi(P)`$ and causes
  80.5 % less extra line loss, because it acts only when the PCC voltage leaves the deadband,
  whereas $`\cos\varphi(P)`$ absorbs whenever output exceeds 50 % of $`P_n`$, whatever the voltage.
- In absolute terms the extra losses are small (9.22 MWh/yr against
  2,878 MWh of annual PV output); the larger difference is the reactive energy the DSO network has to
  supply.

![Sunniest May-June week voltage and Q(U) profile](docs/img/voltage_qu_profile.png)

The figure shows 26 May-1 Jun (the sunniest May-June week of the PVGIS year) for a 3.11 MW plant on F1,
its $`Q(U)`$ hosting capacity: the highest PCC voltage is 1.0618 pu at unity power
factor, 1.0426 pu with $`\cos\varphi(P)`$ and 1.0458 pu with $`Q(U)`$.

### 3. Export caps: what the cap costs in energy

A cap $`c`$ on active export lets a plant of $`P_{inst} = P_{exp} / c`$ share an export limit $`P_{exp}`$. That
oversizing is plain arithmetic ($`1/c`$), not a power-flow result: with a 70 % cap, F1 could take
4.44 MW of PV behind its 3.11 MW $`Q(U)`$ limit and the GPZ busbar 30.77 MW behind
21.54 MW. What the hourly profile adds is the energy cost. PV profile: PVGIS 2023 at 52.1° N, 19.4° E
(1,043.1 kWh/kWp DC), 1,251.3 MWh per MW of AC capacity per year with DC/AC = 1.2, peak 1.000 pu (inverter limit).
Battery: 0.35 $`P_{inst}`$ for 2 h, 90 % round trip, charging only from output above the cap and discharging into
headroom below it.

| Export cap (% of $`P_{inst}`$) | Installed MW per MW of export | Curtailed PV energy | Hours above cap | Not delivered with battery | MWh per MW of export limit | Energy gain vs firm plant | With battery |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 % | 1.000 | 0.00 % | 0 h | 0.00 % | 1,251.3 | +0.0 % | +0.0 % |
| 90 % | 1.111 | 0.56 % | 145 h | 0.06 % | 1,382.6 | +10.5 % | +11.0 % |
| 80 % | 1.250 | 2.51 % | 366 h | 0.28 % | 1,524.9 | +21.9 % | +24.7 % |
| **70 %** | **1.429** | **6.53 %** | **632 h** | **1.62 %** | **1,670.9** | **+33.5 %** | **+40.5 %** |
| 60 % | 1.667 | 12.44 % | 863 h | 5.02 % | 1,826.1 | +45.9 % | +58.3 % |
| 50 % | 2.000 | 20.32 % | 1,103 h | 11.24 % | 1,994.1 | +59.4 % | +77.5 % |

- **A 70 % cap curtails 6.53 % of the PV energy (632 h/yr) and delivers
  33.5 % more MWh through the same export limit than a firm plant** (installed MW +42.9 %).
  The battery cuts the energy not delivered to 1.62 % (round-trip losses included).
- **The weather year matters little, the profile shape a lot.** Over the 19 PVGIS years 2005-2023 the 70 % cap
  curtails 5.55-7.51 % of the energy (median 6.62 %,
  [`results/export_cap_by_pv_year.csv`](results/export_cap_by_pv_year.csv)). A tilted array with DC/AC = 1.2 spends
  many spring and summer hours near its inverter limit; an earlier synthetic profile (solar elevation times daily
  cloudiness) had few such hours and gave 1.78 % (see Changes).
- The trade-off steepens below 70 %: at 50 % the cap curtails 20.32 % of the energy.

![Export cap trade-off](docs/img/curtailment_tradeoff.png)

### 4. GPZ under `N-1` over a year

Hourly AC power flow (8,760 h) of the network with one 25 MVA transformer out, all new generation on the 15 kV
busbar with an export limit equal to the busbar hosting capacity (21.54 MW), existing busbar PV
(8 MW) following the PV profile and loads following the load profile:

| Scenario | New generation behind the 21.54 MW limit | Highest transformer S | Hours above 90 % of 25 MVA | Hours above 25 MVA | Energy exported | Energy not delivered |
|---|---|---:|---:|---:|---:|---:|
| A. Firm PV | 21.54 MW PV | 22.97 MVA | 6 h | 0 h | 26.95 GWh | 0.00 % |
| B. Capped PV | 30.77 MW PV, cap 70 % | 23.66 MVA | 44 h | 0 h | 35.99 GWh | 6.53 % |
| C. Cable pooling | 21.54 MW PV + 10.77 MW wind + 5.38 MW / 10.77 MWh battery | 23.47 MVA | 29 h | 0 h | 53.59 GWh | 0.53 % |

- No scenario overloads the remaining transformer in any hour: the export limit was sized for minimum load with all
  PV at rated output, and in the PVGIS year the hours with PV at its inverter limit never coincide with minimum
  load (midday load stays above its minimum).
- **The same 21.54 MW connection carries 1.99 times the energy
  of the firm PV plant when wind is pooled behind it** (53.59 vs 26.95 GWh/yr).
  Because the synthetic wind and PV rarely peak together, only 1.29 % of the pooled energy
  would be spilled without the battery and 0.53 % with it (conversion losses included).
- The capped plant (B) pushes the transformer close to its limit more often (44 h above 90 % vs
  6 h for A) while exporting 33.5 % more energy.

![GPZ N-1 transformer loading duration curve and seasonal heatmap](docs/img/n1_transformer_loading.png)

## Network data

Conductor data at 20 °C per kilometre (catalogue values, to be verified against the supplier of a real project).
Overhead reactance and susceptance come from the conductor radius and an assumed 1.0 m geometric mean phase spacing;
overhead ampacity from the IEEE Std 738-2012 heat balance at 35 °C ambient, 0.6 m/s wind, 1,000 W/m² sun and the
80 °C continuous limit the manufacturer gives for AFL-6.

| Conductor | $`R`$ (Ω/km) | $`X`$ (Ω/km) | $`B`$ (µS/km) | $`I_{dd}`$ (A) | $`S`$ at 15 kV (MVA) | $`R/X`$ | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| `AFL-6 120` | 0.2356 | 0.3203 | 3.61 | 400 | 10.38 | 0.74 | R, d: Tele-Fonika / Eltrim; X, B: geometry; I: IEEE 738 |
| `AFL-6 70` | 0.4324 | 0.3409 | 3.38 | 271 | 7.04 | 1.27 | R, d: Tele-Fonika / Eltrim; X, B: geometry; I: IEEE 738 |
| `AFL-6 50` | 0.5939 | 0.3512 | 3.27 | 222 | 5.76 | 1.69 | R, d: Tele-Fonika / Eltrim; X, B: geometry; I: IEEE 738 |
| `NA2XS(F)2Y 1x150/25` | 0.2060 | 0.1162 | 78.54 | 320 | 8.31 | 1.77 | NKT data sheet (I: in ground, trefoil) |
| `NA2XS(F)2Y 1x240/25` | 0.1250 | 0.1068 | 91.11 | 420 | 10.91 | 1.17 | NKT data sheet (I: in ground, trefoil) |

Sources: Tele-Fonika Kable, AFL catalogue sheet (PN-EN 50182), and Eltrim Kable, AFL-6 data sheet; NKT,
NA2XS(F)2Y (XUHAKXS) 12/20 kV data sheet (in-ground ratings for trefoil, 0.7 m depth, soil 1.0 K·m/W). The
110/15 kV transformers use typical 25 MVA values ($`u_k = 11.5\%`$, $`P_{cu} = 125\text{ kW}`$,
$`P_{fe} = 22\text{ kW}`$, $`\pm 9 \times 1.78\%`$ OLTC), modelled as a continuous tap that holds the busbar at
$`1.000\text{ pu}`$. Feeder loads are assumed values (minimum and peak per bus).

## Verification

The polar Newton-Raphson `Ybus` solver (`src/grid_ncrfg/powerflow.py`) is checked against the closed-form solution
of a 2-bus line ($`R = 0.088\text{ pu}`$, $`X = 0.0926\text{ pu}`$, $`P_1 = 6.20\text{ MW}`$,
$`Q_1 = -2.04\text{ Mvar}`$, $`S_{base} = 10\text{ MVA}`$, $`V_0 = 1.000\text{ pu}`$):

| Quantity | Closed form | Newton-Raphson `Ybus` (5 iter) | Absolute error |
|---|---:|---:|---:|
| Receiving-end voltage magnitude $`\|V_1\|`$ (pu) | `1.0318964521` | `1.0318964521` | `0.00e+00` |
| Receiving-end voltage angle $`\theta_1`$ (rad) | `0.0730995442` | `0.0730995442` | `9.71e-17` |
| Series loss $`I^2 R`$ (MW) | `0.3520759372` | `0.3520759372` | `5.00e-16` |
| Active power balance $`P_{slack} + P_1 - P_{loss}`$ (MW) | `0` | `1.24e-14` | `1.24e-14` |
| Reactive power balance $`Q_{slack} + Q_1 - Q_{loss}`$ (Mvar) | `0` | `2.66e-15` | `2.66e-15` |

The test suite also checks that the $`Q(U)`$ equilibrium solved inside the Newton iteration matches an independent
outer fixed-point loop, that every reported hosting capacity is feasible and becomes infeasible 50 kW higher, the
battery energy balance, and that `results/`, `docs/data.js` and this README carry the same numbers.

## Method

### Nodal admittance matrix and polar Newton-Raphson AC power flow

Each 15 kV overhead or cable segment between buses $`i`$ and $`j`$ is a $`\pi`$ model with series impedance
$`Z_{ij} = R_{ij} + j X_{ij}`$ and shunt susceptance $`B_{ij}`$ in per unit on $`S_{base} = 10\text{ MVA}`$ and
$`U_{base} = 15\text{ kV}`$ ($`Z_{base} = 22.5\ \Omega`$). The 110/15.75 kV transformers carry an off-nominal tap
ratio $`t:1`$ on the HV side; `N-1` removes one of the two identical parallel units (series impedance doubles,
rating 25 MVA). For bus voltages $`V_i \angle \theta_i`$ and $`Y_{ik} = G_{ik} + j B_{ik}`$:

```math
\begin{aligned}
P_i(\theta, V) &= V_i \sum_{k=0}^{N-1} V_k \left( G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik} \right) \\
Q_i(\theta, V) &= V_i \sum_{k=0}^{N-1} V_k \left( G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik} \right)
\end{aligned}
```

where $`\theta_{ik} = \theta_i - \theta_k`$. The polar Jacobian $`H = \partial P / \partial \theta`$,
$`N = \partial P / \partial V`$, $`M = \partial Q / \partial \theta`$, $`L = \partial Q / \partial V`$ is assembled
analytically over the PQ buses:

```math
\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} =
\begin{bmatrix} H & N \\ M & L - \mathrm{diag}\left(\dfrac{\partial Q_{spec}}{\partial V}\right) \end{bmatrix}
\begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}, \qquad
\max\left(\|\Delta P\|_\infty, \|\Delta Q\|_\infty\right) < 10^{-11}\text{ pu}
```

The extra diagonal term is non-zero only at buses with a voltage-dependent reactive set point ($`Q(U)`$), so the
controller equilibrium is solved together with the power flow.

### Closed-form 2-bus benchmark

For a 2-bus line with slack voltage $`V_0 \angle 0`$ and injection $`S_1 = P_1 + j Q_1`$ through $`Z = R + j X`$,
eliminating $`\theta_1`$ gives a biquadratic equation in $`V_1^2`$:

```math
V_1^4 - \left[ V_0^2 + 2(R P_1 + X Q_1) \right] V_1^2 + (R^2 + X^2)(P_1^2 + Q_1^2) = 0
```

Linearising around $`V_1 \approx V_0 = 1\text{ pu}`$ gives $`\Delta U \approx R P_1 + X Q_1`$: absorbing
$`Q_1 = -0.33\,P_1`$ cancels $`0.33 / (R/X)`$ of the active voltage rise, about 25 % on F1
($`R/X = 1.33`$).

### Reactive power control modes

$`\cos\varphi(P)`$: $`\cos\varphi = 1`$ up to $`0.5\,P_n`$, then falling linearly with $`P`$ to 0.95 inductive at
$`P_n`$, $`Q = -P \tan(\arccos\cos\varphi(P))`$. $`Q(U)`$ droop at each plant bus:

```math
Q(U) =
\begin{cases}
+Q_{\max} & U \le 0.95\text{ pu} \\[1ex]
+Q_{\max}\,\dfrac{0.98 - U}{0.98 - 0.95} & 0.95\text{ pu} < U < 0.98\text{ pu} \\[2ex]
0 & 0.98\text{ pu} \le U \le 1.02\text{ pu} \\[2ex]
-Q_{\max}\,\dfrac{U - 1.02}{1.05 - 1.02} & 1.02\text{ pu} < U < 1.05\text{ pu} \\[1ex]
-Q_{\max} & U \ge 1.05\text{ pu}
\end{cases}
\qquad Q_{\max} = 0.33\,P_n
```

with $`Q_{\max}`$ reduced proportionally when available power is below $`0.1\,P_n`$. These parameters, the
$`1.05\text{ pu}`$ planning limit and the 50 % knee are study assumptions. Which article of Commission Regulation
(EU) 2016/631 (NC RfG) applies depends on the plant type. In Poland a power park module connected below 110 kV is
type B from 0.2 MW and type C from 10 MW (type D from 75 MW or at 110 kV and above; thresholds set by URE in 2018,
to be verified). The feeder plants on F1 and F2 (2.30-6.31 MW, also 4.44 MW behind a 70 % cap) are type B:
Art. 20(2)(a) lets the system operator specify their reactive power capability. The F3 plant (10.77-11.41 MW) and the
GPZ busbar plants (19.59-30.77 MW) are type C: Art. 21(3)(b)-(d) add the reactive power capability profiles and the
voltage, reactive power and power factor control modes. The settings themselves come from the Polish requirements of
general application adopted under the regulation, the DSO grid code (IRiESD) and the connection conditions of each
plant; they change over time and must be verified for any real case.

### Hosting capacity, battery and profiles

- Hosting capacity: 18-step bisection on installed MW over the full AC power flow; feasible means every feeder bus
  $`\le 1.05\text{ pu}`$ and every feeder branch $`\le 100\%`$ of ampacity (GPZ: also the remaining transformer
  $`\le 25\text{ MVA}`$). The binding constraint is the one violated just above the reported value.
- Battery: hourly dispatch behind a shared export limit; charge from output above the limit, discharge into headroom
  below it, power, energy and $`\sqrt{0.9}`$ charge/discharge efficiency limits (`dispatch_export_limited_bess`).
- Profiles (`src/grid_ncrfg/profiles.py`): PV is the hourly PVGIS 5.3 output (PVGIS-SARAH3 radiation, ERA5
  temperature) at 52.1° N, 19.4° E for a fixed south-facing c-Si array at the PVGIS optimal 40° tilt with 14 % losses,
  2005-2023 (stored in `src/grid_ncrfg/data/`); the study year is 2023, the most recent complete year, laid on the
  2026 calendar and converted to AC with DC/AC = 1.2. Synthetic and seeded: wind from an AR(1) Weibull wind speed and a generic 3-12-25 m/s
  power curve scaled to 2,500 full-load hours (assumed, of the order of the Polish onshore fleet; to be
  verified); load from a seasonal and daily shape between 33 % and 100 % of peak.

## Limitations

- **Synthetic network, load and wind.** Feeder lengths, loads, the existing 8 MW of busbar PV
  and the load and wind profiles are assumptions; only conductor data and the PV series come from external sources.
  The PV series is one site and one array orientation; east-west or tracking arrays reach the inverter limit in
  fewer hours.
- **Balanced positive-sequence steady state.** Unbalance, earth faults (Petersen coil compensation typical of Polish
  15 kV networks), harmonics and dynamics are not modelled.
- **No short-circuit or protection study.** Real connection studies also check fault levels against switchgear
  ratings and protection selectivity.
- **Planning limits.** The voltage limit is a single $`1.05\text{ pu}`$ value at minimum load; DSOs use their own
  criteria, operating ranges and load assumptions. Resistances are at 20 °C (no temperature correction) and cable
  ratings assume one installation condition.
- **Cable pooling and export caps.** The legal framework (cable pooling was added to the Polish Energy Law,
  *Prawo energetyczne*, in 2023; conditions of non-firm connection agreements) is not modelled and should be checked
  in the current law and DSO practice. The battery policy minimises spilled energy only; it does not trade.

## Data

| Data | Source | Terms |
|---|---|---|
| Hourly PV output 2005-2023, 52.1° N 19.4° E | [PVGIS 5.3](https://re.jrc.ec.europa.eu/pvg_tools/en/), European Commission, Joint Research Centre (PVGIS-SARAH3 radiation, ERA5 temperature) | © European Union; reuse with acknowledgement of the source (Commission Decision 2011/833/EU) |
| Conductor data | Tele-Fonika Kable, Eltrim Kable, NKT data sheets | catalogue values, see Network data |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"          # Linux/macOS: .venv/bin/pip
pytest -q                                      # offline tests
python -m grid_ncrfg verify                    # 2-bus closed form vs Newton-Raphson Ybus
python -m grid_ncrfg run                       # results/, docs/img/ and docs/ dashboard (a few minutes)
```

Repository layout:

```
src/grid_ncrfg/
  network.py     GPZ 110/15 kV substation, AFL-6 / NA2XS(F)2Y line data, 4 radial feeders, Ybus builder
  powerflow.py   polar Newton-Raphson AC power flow (with optional Q(U) buses) and 2-bus closed form
  ncrfg.py       cos-phi(P) and Q(U) control laws, feeder plant power flow
  profiles.py    8,760-hour synthetic PV, wind and MV load profiles
  hosting.py     hosting capacity search, export-cap trade-off, battery dispatch, GPZ N-1 time series
  pipeline.py    end-to-end runner, Matplotlib charts, static Plotly dashboard generator
  cli.py         command-line entry point
tests/           pytest suite (invariants and output consistency)
results/         CSV and JSON study outputs
docs/            GitHub Pages dashboard and figures
```

## Changes

- 2026-10-31: the PV profile is now the hourly PVGIS series instead of a synthetic one (solar elevation times
  random daily cloudiness, scaled to the PVGIS annual yield). The synthetic shape had too few hours near the
  inverter limit: a 70 % export cap curtailed 1.78 % of the energy (238 h) with it, against 6.53 % (632 h) with
  PVGIS 2023. The reactive-energy, export-cap and `N-1` results changed accordingly; the hosting capacities do not
  depend on the profile. The NC RfG reference now distinguishes type B (Art. 20) and type C (Art. 21) plants.

MIT licence.
