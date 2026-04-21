"""
Simple residential DHW loop with recirculation pump — flow solver.

Topology
--------
::

    Boiler → pipe0 (supply) → Shower  [draw]
           → pipe1           → Faucet  [draw]
           → pipe2           → Sink    [draw]
           → pipe3 (return)  → Pump → Boiler

    Cold makeup enters at the boiler and equals Σ draw rates (global
    mass balance).  The recirculation pump on pipe 3 imposes a constant
    ṁ_recirc.

Draw profiles
-------------
Draw profiles are stored as CSV files (columns ``time_s, mdot_kg_s``).
Use :func:`generate_draw_profile_csv` to create them and
:func:`csv_draw_profile` to wrap a CSV as a callable ``f(t) -> kg/s``
suitable for :class:`dhw.models.DrawNode.draw_profile`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from dhw import DHWNetwork, DrawNode, PipeParams
from dhw.network import _cell_at, default_pipe  # noqa: F401  (re-exported helpers)


# ──────────────────────────────────────────────────────────────────────
# Draw-profile CSV helpers
# ──────────────────────────────────────────────────────────────────────

def generate_draw_profile_csv(
    path: str | Path,
    peak_kg_s: float,
    on_windows_hours: Sequence[tuple[float, float]],
    dt_s: float = 60.0,
    t_end_s: float = 86400.0,
) -> Path:
    """Generate a step-function draw profile CSV over a 24-hour day.

    Parameters
    ----------
    path :
        Output CSV path. Parent directories are created automatically.
    peak_kg_s :
        Mass flow rate (kg/s) during "on" windows.
    on_windows_hours :
        Iterable of ``(start_hour, end_hour)`` tuples (0–24 h) describing
        the periods when the draw is active.
    dt_s :
        Sampling period written to the CSV. Defaults to 60 seconds.
    t_end_s :
        End time of the profile, in seconds. Defaults to 86400 s (24 h).

    Returns
    -------
    pathlib.Path
        The path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.arange(0.0, t_end_s + dt_s, dt_s)
    mdot = np.zeros_like(times)
    for h_start, h_end in on_windows_hours:
        t0, t1 = h_start * 3600.0, h_end * 3600.0
        mdot[(times >= t0) & (times < t1)] = peak_kg_s

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "mdot_kg_s"])
        for t, m in zip(times, mdot):
            writer.writerow([f"{t:.3f}", f"{m:.6f}"])

    return path


def csv_draw_profile(
    path: str | Path,
    period_s: float | None = 86400.0,
) -> Callable[[float], float]:
    """Wrap a CSV draw profile as a callable ``f(t_seconds) -> kg/s``.

    The CSV is loaded once at wrap time; subsequent calls are O(log N)
    (``np.searchsorted``).  A step-function interpretation is used: the
    returned value at time ``t`` is the ``mdot`` of the latest sample
    whose timestamp is ≤ ``t``.

    Parameters
    ----------
    path :
        Path to a CSV with columns ``time_s`` and ``mdot_kg_s``.
    period_s :
        If not ``None``, ``t`` is wrapped modulo ``period_s`` so that the
        profile repeats indefinitely (useful for multi-day simulations
        driven by a single 24-hour CSV).

    Returns
    -------
    Callable
        ``f(t) -> mdot`` where ``t`` is seconds and ``mdot`` is kg/s.
    """
    path = Path(path)
    times: list[float] = []
    flows: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "time_s" not in reader.fieldnames \
                or "mdot_kg_s" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have columns 'time_s' and 'mdot_kg_s'; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            times.append(float(row["time_s"]))
            flows.append(float(row["mdot_kg_s"]))

    t_arr = np.asarray(times, dtype=float)
    q_arr = np.asarray(flows, dtype=float)

    if t_arr.size == 0:
        raise ValueError(f"{path} contains no samples")
    if np.any(np.diff(t_arr) < 0):
        raise ValueError(f"{path} timestamps must be monotonically non-decreasing")

    def profile(t: float) -> float:
        tt = t % period_s if period_s is not None else t
        idx = int(np.searchsorted(t_arr, tt, side="right") - 1)
        if idx < 0:
            idx = 0
        return float(q_arr[idx])

    profile.__name__ = f"csv_draw_profile[{path.name}]"
    return profile


# ──────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────────────────

def _print_discretisation(net: DHWNetwork, dz: float) -> None:
    print(f"  Target dz = {dz} m — pipe discretisation:")
    for name, pipe in zip(net.pipe_names, net.pipes):
        actual_dz = pipe.length / pipe.N_cells
        print(f"    {name:<18s}  L={pipe.length:5.1f} m  "
              f"N={pipe.N_cells:3d} cells  dz={actual_dz:.3f} m")


def _print_network_diagram(net: DHWNetwork) -> None:
    print("\n  Network diagram:")
    print("    Boiler")
    print("      |")
    print("      +-- pipe0 --> Shower (draw)")
    print("                      |")
    print("                      +-- pipe1 --> Faucet (draw)")
    print("                                      |")
    print("                                      +-- pipe2 --> Sink (draw)")
    print("                                                      |")
    print("                                                      +-- pipe3 --> Pump --> Boiler")
    print(f"    Recirculation setpoint: mdot_recirc = {net.mdot_recirc:g} kg/s")


# ──────────────────────────────────────────────────────────────────────
# Hydraulic model (mass balance)
# ──────────────────────────────────────────────────────────────────────

def _residential_pipe_flow(net: DHWNetwork, draw_at_pipe_cell: dict) -> np.ndarray:
    """Solve pipe flow rates for the 4-pipe residential loop.

    Topology::

        Boiler --pipe0--> [Shower] --pipe1--> [Faucet] --pipe2--> [Sink]
               --pipe3--> Pump --> Boiler

    Known:
        - ṁ_pump  = mdot_recirc  (fixed by pump, flows in pipe 3)
        - ṁ_draw_shower, ṁ_draw_faucet, ṁ_draw_sink  (time-varying)

    Node mass balance (no storage in pipes)::

        Pump node  :  ṁ3 = ṁ_pump                       = mdot_recirc
        Sink node  :  ṁ2 = ṁ3 + ṁ_sink
        Faucet     :  ṁ1 = ṁ2 + ṁ_faucet
        Shower     :  ṁ0 = ṁ1 + ṁ_shower
        Boiler     :  ṁ_cold = ṁ0 − ṁ3 = Σ draws
    """
    cpc = [p.N_cells for p in net.pipes]
    q = np.zeros(len(net.pipes))

    mdot_shower = draw_at_pipe_cell.get((0, cpc[0] - 1), 0.0)
    mdot_faucet = draw_at_pipe_cell.get((1, cpc[1] - 1), 0.0)
    mdot_sink   = draw_at_pipe_cell.get((2, cpc[2] - 1), 0.0)

    q[3] = net.mdot_recirc
    q[2] = q[3] + mdot_sink
    q[1] = q[2] + mdot_faucet
    q[0] = q[1] + mdot_shower

    return q


# ──────────────────────────────────────────────────────────────────────
# Network builder
# ──────────────────────────────────────────────────────────────────────

def build_residential_network(
    dz: float = 0.5,
    profiles_dir: str | Path = "output/data/draw_profiles",
) -> DHWNetwork:
    """Build the 4-pipe residential DHW loop with a recirculation pump.

    Draw profiles for the three fixtures (shower, faucet, sink) are
    generated as CSV files under ``profiles_dir`` and wired into each
    :class:`dhw.models.DrawNode` via :func:`csv_draw_profile`.

    Pipe index map
    --------------
    ===  ===================  ======  =========================================
    #    name                 length  segment
    ===  ===================  ======  =========================================
    0    boiler_to_shower     6 m     Boiler → Shower
    1    shower_to_faucet     3 m     Shower → Faucet
    2    faucet_to_sink       4 m     Faucet → Sink
    3    sink_to_boiler       10 m    Sink → Pump → Boiler (return)
    ===  ===================  ======  =========================================

    Parameters
    ----------
    dz :
        Target finite-volume cell size, in metres.
    profiles_dir :
        Directory where generated draw-profile CSVs are written. Any
        existing files with the same names are overwritten.

    Returns
    -------
    DHWNetwork
        Fully configured residential network.
    """
    profiles_dir = Path(profiles_dir)

    net = DHWNetwork(
        T_supply=60.0,
        heater_power=4500.0,
        boiler_volume=0.189,
        mdot_recirc=0.03,
        T_ambient=16.0,
        T_cold=15.0,
        return_pipe_idx=3,
        pipe_flow_fn=_residential_pipe_flow,
        dz=dz,
    )
    net.heater_efficiency = 0.90

    pipe_lengths = [6.0, 3.0, 4.0, 10.0]
    net.pipes = [default_pipe(length=L, dz=dz) for L in pipe_lengths]
    net.pipes[3] = default_pipe(length=10.0, dz=dz, d_inner=0.015, insul_t=0.02)

    net.pipe_names = [
        "boiler_to_shower",
        "shower_to_faucet",
        "faucet_to_sink",
        "sink_to_boiler",
    ]
    net.parent_map = {1: 0, 2: 1, 3: 2}
    net.adjacency = [(0, 1), (1, 2), (2, 3)]

    # ── Generate per-fixture draw-profile CSVs ──
    # Shower: ~10 min morning, ~8 min evening
    shower_csv = generate_draw_profile_csv(
        profiles_dir / "shower.csv",
        peak_kg_s=0.15,
        on_windows_hours=[(6.75, 6.92), (21.5, 21.63)],
    )

    # Bathroom faucet: handwashing/teeth before & after shower, midday, evening
    faucet_csv = generate_draw_profile_csv(
        profiles_dir / "faucet.csv",
        peak_kg_s=0.05,
        on_windows_hours=[
            (6.50, 6.53),   # pre-shower teeth brushing ~2 min
            (6.95, 6.98),   # post-shower handwash ~2 min
            (7.50, 7.52),   # quick handwash before leaving
            (12.25, 12.28), # midday handwash
            (17.00, 17.02), # arrive home handwash
            (21.40, 21.43), # pre-shower
            (21.67, 21.72), # post-shower face wash ~3 min
            (22.50, 22.55), # teeth brushing before bed ~3 min
        ],
    )

    # Kitchen sink: morning dishes/coffee, dinner prep & cleanup
    sink_csv = generate_draw_profile_csv(
        profiles_dir / "sink.csv",
        peak_kg_s=0.10,
        on_windows_hours=[
            (6.50, 6.58),   # morning coffee/rinse ~5 min
            (7.00, 7.05),   # breakfast dishes ~3 min
            (12.00, 12.08), # lunch prep ~5 min
            (18.00, 18.17), # dinner prep ~10 min
            (18.75, 19.00), # dinner cleanup ~15 min
            (21.00, 21.05), # evening snack cleanup ~3 min
        ],
    )

    cpc = [p.N_cells for p in net.pipes]
    net.draw_nodes = [
        DrawNode("Shower", 0, cpc[0] - 1, csv_draw_profile(shower_csv)),
        DrawNode("Faucet", 1, cpc[1] - 1, csv_draw_profile(faucet_csv)),
        DrawNode("Sink",   2, cpc[2] - 1, csv_draw_profile(sink_csv)),
    ]

    _print_discretisation(net, dz)
    return net


# ──────────────────────────────────────────────────────────────────────
# Pipe heat loss calculator
# ──────────────────────────────────────────────────────────────────────

def compute_pipe_UA(p: PipeParams) -> float:
    """
    Overall heat loss coefficient per unit length [W/(m·K)].

    Uses the radial conduction + external convection model (Klimczak et al., Eq. 2).
    Internal convection resistance is neglected (turbulent flow → h_in → ∞).
    """
    d_i = p.inner_diameter
    d_o = d_i + 2 * p.wall_thickness
    d_ins = d_o + 2 * p.insulation_t

    R_pipe = np.log(d_o / d_i) / (2 * np.pi * p.pipe_k)
    R_ins  = np.log(d_ins / d_o) / (2 * np.pi * p.insulation_k)
    R_ext  = 1.0 / (np.pi * d_ins * p.h_ext)

    return 1.0 / (R_pipe + R_ins + R_ext)   # W/(m·K)


def pipe_downstream_temp(
    pipe: PipeParams,
    flow_rate: float,
    T_upstream: float,
    dt: float,
    T_ambient: float,
    rho: float = 998.0,
    c: float = 4186.0,
) -> float:
    """Compute the outlet temperature of a pipe after one timestep *dt*.

    Solves the 1-D advection–heat-loss PDE on the pipe's ``N_cells`` finite-
    volume cells using an explicit first-order upwind scheme.  The CFL and
    Fourier stability constraints are enforced automatically via sub-stepping.

    Governing equation (per cell *i*):

    .. math::

        \\rho A \\Delta z \\, c \\frac{\\partial T_i}{\\partial t}
        = \\dot{m} c (T_{i-1} - T_i)
          - U\\!A_L \\Delta z (T_i - T_{\\text{amb}})

    where :math:`T_{-1} = T_{\\text{upstream}}` is the inlet boundary condition
    and the cell temperature profile is initialised uniformly at *T_upstream*.

    Parameters
    ----------
    pipe :
        Pipe geometry and thermal properties (already discretised into
        ``pipe.N_cells`` cells of length ``dz = pipe.length / pipe.N_cells``).
    flow_rate :
        Mass flow rate through the pipe [kg/s].  Zero flow is allowed
        (pure conduction / heat-loss, no advection).
    T_upstream :
        Temperature at the upstream (inlet) node [°C].
    dt :
        Macro-timestep over which to advance the solution [s].
    T_ambient :
        Ambient temperature surrounding the pipe [°C].
    rho :
        Water density [kg/m³].  Defaults to 998 kg/m³.
    c :
        Specific heat of water [J/(kg·K)].  Defaults to 4186 J/(kg·K).

    Returns
    -------
    float
        Temperature at the downstream (outlet) node after *dt* seconds [°C].
    """
    N   = pipe.N_cells
    dz  = pipe.dz
    A   = np.pi / 4.0 * pipe.inner_diameter ** 2   # m²
    UA_L = compute_pipe_UA(pipe)                    # W/(m·K)

    # ── Stable sub-timestep (explicit scheme) ─────────────────────────
    u = flow_rate / (rho * A) if flow_rate > 0.0 else 0.0
    dt_cfl   = dz / u          if u    > 0.0 else dt  # CFL = 1
    dt_four  = 0.5 * rho * A * c / UA_L if UA_L > 0.0 else dt  # Fourier ≤ 0.5
    dt_sub   = min(dt_cfl, dt_four, dt)
    n_sub    = max(1, int(np.ceil(dt / dt_sub)))
    dt_sub   = dt / n_sub                             # rebalance evenly

    # ── Initial condition: pipe uniformly at upstream temperature ──────
    T = np.full(N, T_upstream, dtype=float)

    for _ in range(n_sub):
        # Upwind inlet: ghost value for cell 0 is the inlet BC
        T_in    = np.empty(N)
        T_in[0] = T_upstream
        T_in[1:] = T[:-1]

        # Explicit upwind update
        T = (T
             - (flow_rate * dt_sub) / (rho * A * dz) * (T - T_in)   # advection
             - (UA_L * dt_sub) / (rho * A * c) * (T - T_ambient))    # heat loss

    return float(T[-1])


# ──────────────────────────────────────────────────────────────────────
# Flow-rate solver over a simulation window
# ──────────────────────────────────────────────────────────────────────

class FlowSolver:
    
    
    def __init__(
        self,
        net: DHWNetwork,
        t_start: float,
        t_end: float,
        dt: float = 1.0,
    ):
        self.net = net
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
    
    def solve_flow_rates(
        self,
    ) -> dict:
        """Compute pipe flow rates at every timestep

        Returns
        -------
        dict
            ``time``       : 1-D array of time values ``(N_steps,)``.
            ``pipe_flows`` : 2-D array ``(N_steps, N_pipes)`` — ṁ in each pipe.
            ``pipe_contributions`` : 3-D array ``(N_steps, N_pipes, N_components)``
                with stacked positive flow contributions in each pipe.
                Component order is ``[recirculation, *draw_nodes]``.
            ``pipe_contribution_labels`` : list of component labels matching the
                third axis of ``pipe_contributions``.
            ``mdot_cold``  : 1-D array ``(N_steps,)`` — cold-makeup mass flow.
            ``draw_rates`` : 2-D array ``(N_steps, N_draws)`` — per-draw rate.
        """
        times = np.arange(self.t_start, self.t_end, self.dt)
        n_steps = len(times)
        n_pipes = len(self.net.pipes)
        n_draws = len(self.net.draw_nodes)


        pipe_flows = np.zeros((n_steps, n_pipes))
        draw_rates = np.zeros((n_steps, n_draws))
        mdot_cold = np.zeros(n_steps)

        for k, t in enumerate(times):
            draw_at_pipe_cell: dict[tuple[int, int], float] = {}
            for j, dn in enumerate(self.net.draw_nodes):
                mdot = dn.draw_profile(t)
                draw_rates[k, j] = mdot
                draw_at_pipe_cell[(dn.pipe_index, dn.cell_index)] = mdot

            q = self.net.pipe_flow_fn(self.net, draw_at_pipe_cell)
            pipe_flows[k, :] = q
            mdot_cold[k] = draw_rates[k, :].sum()

        # Build a stackable decomposition of each pipe flow:
        # q_pipe = recirculation contribution + downstream draw contributions.
        contribution_labels = ["recirculation", *[dn.name for dn in self.net.draw_nodes]]
        n_components = len(contribution_labels)
        pipe_contributions = np.zeros((n_steps, n_pipes, n_components))

        # Recirculation contribution is present in every pipe in this loop model.
        pipe_contributions[:, :, 0] = self.net.mdot_recirc

        # A draw contributes to its own pipe and all upstream supply pipes.
        # We determine upstream pipes through parent_map ancestry.
        for j, dn in enumerate(self.net.draw_nodes):
            upstream = set()
            p = dn.pipe_index
            while True:
                upstream.add(p)
                if p not in self.net.parent_map:
                    break
                p = self.net.parent_map[p]
            for p_idx in upstream:
                pipe_contributions[:, p_idx, j + 1] = draw_rates[:, j]

        return {
            "time": times,
            "pipe_flows": pipe_flows,
            "pipe_contributions": pipe_contributions,
            "pipe_contribution_labels": contribution_labels,
            "mdot_cold": mdot_cold,
            "draw_rates": draw_rates,
        }

    def simulate_temperatures(self, T_init: float) -> dict:
        """Simulate water temperature across the network over the full time window.

        Follows the same two-step pattern as :meth:`solve_flow_rates`:

        1. **Flow pass** — call :meth:`solve_flow_rates` to get the mass-flow
           rate in every pipe at every timestep.
        2. **Temperature pass** — integrate the coupled thermal ODE system
           using those pre-computed flow rates as a zero-order-hold (step)
           input signal looked up inside the RHS.

        The coupled ODE system is:

        * **Boiler** (well-mixed tank):

          .. math::

              \\rho V c \\frac{dT_b}{dt}
              = Q_{\\text{htr}}
              + \\dot{m}_{\\text{cold}} c (T_{\\text{cold}} - T_b)
              + \\dot{m}_{\\text{recirc}} c (T_{\\text{ret}} - T_b)

        * **Each pipe cell** *j* of pipe *i* (upwind advection + wall loss + optional HX):

          .. math::

              \\rho A_i \\Delta z_i c \\frac{dT_{i,j}}{dt}
              = \\dot{m}_i c (T_{i,j-1} - T_{i,j})
              - U\\!A_i \\Delta z_i (T_{i,j} - T_{\\text{amb}})
              - U\\!A_{\\text{hx}} (T_{i,j} - T_{\\text{proc}})

        ``scipy.integrate.solve_ivp`` with ``LSODA`` handles the nonlinear
        thermostat and auto-detects stiffness from cells with small time
        constants.

        Returns
        -------
        dict
            ``time``        : 1-D array ``(N_t,)`` of evaluation times [s].
            ``T_boiler``    : 1-D array ``(N_t,)`` — boiler temperature [°C].
            ``T_draw``      : 2-D array ``(N_t, N_draws)`` — temperature at
                             each draw node [°C].
            ``draw_names``  : list of draw-node name strings.
            ``T_pipe``      : dict ``{pipe_idx: ndarray (N_t, N_cells)}`` —
                             full spatial temperature history of every pipe.
            ``pipe_flows``  : 2-D array ``(N_t, N_pipes)`` — pre-computed
                             mass-flow rates from the flow pass.
            ``heater_power_input_W`` : 1-D array ``(N_t,)`` — electric
                            heater input power [W].
            ``heater_power_to_water_W`` : 1-D array ``(N_t,)`` — thermal
                            power delivered to water [W].
            ``heater_energy_input_kWh`` : 1-D array ``(N_t,)`` — cumulative
                            electric energy [kWh].
            ``heater_energy_to_water_kWh`` : 1-D array ``(N_t,)`` —
                            cumulative thermal energy to water [kWh].
        """
        # ── Step 1: hydraulic pass ─────────────────────────────────────
        flow_result = self.solve_flow_rates()
        times_flow  = flow_result["time"]           # (N_steps,)
        pipe_flows  = flow_result["pipe_flows"]     # (N_steps, N_pipes)
        mdot_cold_t = flow_result["mdot_cold"]      # (N_steps,)  Σ draws

        net     = self.net
        pipes   = net.pipes
        n_pipes = len(pipes)

        # ── State-vector layout ────────────────────────────────────────
        # y[0]                            = T_boiler
        # y[offsets[i] : offsets[i]+N_i] = T_pipe_i   (i = 0 … n_pipes-1)
        offsets = np.zeros(n_pipes, dtype=int)
        offsets[0] = 1
        for i in range(1, n_pipes):
            offsets[i] = offsets[i - 1] + pipes[i - 1].N_cells
        n_states = 1 + sum(p.N_cells for p in pipes)

        # ── Pre-computed per-pipe geometry ─────────────────────────────
        pipe_A   = np.array([np.pi / 4.0 * p.inner_diameter ** 2 for p in pipes])
        pipe_UA  = np.array([compute_pipe_UA(p) for p in pipes])
        pipe_dz  = np.array([p.dz for p in pipes])

        # ── Step 2: thermal ODE pass ───────────────────────────────────
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            "The function to integrate the thermal ODE system."
            """
            Parameters
            ----------
            t : float
                The current time.
            y : np.ndarray
                The current state vector.

            Returns
            -------
            np.ndarray
                The derivative of the state vector.
            """
            T_boiler = y[0]
            dy       = np.zeros(n_states)

            # Look up pre-computed flow rates (zero-order hold on the
            # hydraulic grid — same step-function semantics as draw profiles)
            k    = max(0, int(np.searchsorted(times_flow, t, side="right")) - 1)
            q    = pipe_flows[k]            # (n_pipes,) mass-flow rates [kg/s]
            mdot_cold = mdot_cold_t[k]

            # ── Pipe cell ODEs ─────────────────────────────────────────
            for i, pipe in enumerate(pipes):
                N    = pipe.N_cells
                s    = offsets[i]
                T_p  = y[s : s + N]
                A    = pipe_A[i]
                UA_L = pipe_UA[i]
                mdot = q[i]
                m_cell = net.rho * A * pipe_dz[i]  # kg — thermal mass per cell

                # Upstream BC: boiler outlet or parent pipe's last cell
                if i in net.parent_map:
                    p_idx = net.parent_map[i]
                    T_in  = y[offsets[p_idx] + pipes[p_idx].N_cells - 1]
                else:
                    T_in = T_boiler

                T_ghost     = np.empty(N)
                T_ghost[0]  = T_in
                T_ghost[1:] = T_p[:-1]

                # Advection (upwind) + wall heat loss
                dT = (
                    mdot / m_cell * (T_ghost - T_p)                          # advection  [K/s]
                    - UA_L / (net.rho * A * net.c) * (T_p - net.T_ambient)  # wall loss  [K/s]
                )

                dy[s : s + N] = dT

            # ── Boiler ODE ─────────────────────────────────────────────
            ret_idx     = net.return_pipe_idx
            T_recirc    = y[offsets[ret_idx] + pipes[ret_idx].N_cells - 1]
            Q_htr_input = net.heater_power if T_boiler < net.T_supply else 0.0  # W electric input
            Q_htr       = net.heater_efficiency * Q_htr_input                    # W delivered to water
            M           = net.rho * net.boiler_volume

            dy[0] = (
                Q_htr / (M * net.c)
                + mdot_cold       / M * (net.T_cold - T_boiler)
                + net.mdot_recirc / M * (T_recirc   - T_boiler)
            )

            return dy

        y0  = np.full(n_states, T_init)
        sol = solve_ivp(
            rhs,
            (self.t_start, self.t_end),
            y0,
            method="LSODA",
            t_eval=times_flow,   # same grid as the hydraulic pass
            rtol=1e-3,
            atol=1e-4,
        )

        if not sol.success:
            raise RuntimeError(f"simulate_temperatures: solve_ivp failed — {sol.message}")

        # ── Unpack results ─────────────────────────────────────────────
        T_boiler = sol.y[0]
        heater_on = T_boiler < net.T_supply
        Q_htr_input = np.where(heater_on, net.heater_power, 0.0)
        Q_htr_to_water = net.heater_efficiency * Q_htr_input

        E_htr_input_J = np.zeros_like(sol.t)
        E_htr_to_water_J = np.zeros_like(sol.t)
        E_htr_input_J[1:] = np.cumsum(
            0.5 * (Q_htr_input[:-1] + Q_htr_input[1:]) * np.diff(sol.t)
        )
        E_htr_to_water_J[1:] = np.cumsum(
            0.5 * (Q_htr_to_water[:-1] + Q_htr_to_water[1:]) * np.diff(sol.t)
        )

        n_draws = len(net.draw_nodes)
        T_draw  = np.zeros((len(sol.t), n_draws))
        for j, dn in enumerate(net.draw_nodes):
            T_draw[:, j] = sol.y[offsets[dn.pipe_index] + dn.cell_index]

        T_pipe_out: dict[int, np.ndarray] = {
            i: sol.y[offsets[i] : offsets[i] + pipes[i].N_cells].T
            for i in range(n_pipes)
        }

        return {
            "time":       sol.t,
            "T_boiler":   T_boiler,
            "T_draw":     T_draw,
            "draw_names": [dn.name for dn in net.draw_nodes],
            "T_pipe":     T_pipe_out,
            "pipe_flows": pipe_flows,
            "heater_power_input_W": Q_htr_input,
            "heater_power_to_water_W": Q_htr_to_water,
            "heater_energy_input_J": E_htr_input_J,
            "heater_energy_input_kWh": E_htr_input_J / 3.6e6,
            "heater_energy_to_water_J": E_htr_to_water_J,
            "heater_energy_to_water_kWh": E_htr_to_water_J / 3.6e6,
        }

    def compute_heater_energy(self, result: dict) -> dict:
        """Integrate heater energy from temperature timeseries.
        
        Returns dict with time array, instantaneous power [W], 
        and cumulative energy [J] and [kWh].
        """
        t = result["time"]  # s
        Q_htr = result.get("heater_power_input_W")
        E_cumulative = result.get("heater_energy_input_J")

        if Q_htr is None or E_cumulative is None:
            T_boiler = result["T_boiler"]  # °C
            Q_htr = np.where(T_boiler < self.net.T_supply, self.net.heater_power, 0.0)

            # Backward-compatible fallback when older results don't store heater series.
            E_cumulative = np.zeros_like(t)
            E_cumulative[1:] = np.cumsum(
                0.5 * (Q_htr[:-1] + Q_htr[1:]) * np.diff(t)
            )
        
        return {
            "time": t,
            "power_W": Q_htr,
            "energy_J": E_cumulative,
            "energy_kWh": E_cumulative / 3.6e6,
            "total_kWh": E_cumulative[-1] / 3.6e6,
        }
# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────

def plot_temperature_timeseries(
    result: dict,
    *,
    time_unit: str = "h",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot boiler and per-draw-node temperatures on a single axes.

    Parameters
    ----------
    result :
        Return value of :meth:`FlowSolver.simulate_temperatures`.
    time_unit :
        ``"h"`` (hours) or ``"s"`` (seconds) for the horizontal axis.
    figsize :
        Optional ``(width, height)`` in inches.
    save_path :
        If set, the figure is saved here (parent dirs created automatically).
    show :
        If ``True``, call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_s  = np.asarray(result["time"], dtype=float)
    t_plot, xlabel = (t_s / 3600.0, "Time [h]") if time_unit == "h" else (t_s, "Time [s]")

    T_boiler   = np.asarray(result["T_boiler"])
    T_draw     = np.asarray(result["T_draw"])          # (N_t, N_draws)
    draw_names = list(result["draw_names"])
    n_draws    = T_draw.shape[1]

    w, h = figsize if figsize is not None else (12.0, 5.5)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # Boiler temperature
    ax.plot(t_plot, T_boiler, color="C0", linewidth=1.2, label="Boiler")

    # Per-draw temperatures
    for j in range(n_draws):
        ax.plot(
            t_plot,
            T_draw[:, j],
            color=f"C{j + 1}",
            linewidth=1.2,
            label=f"Draw: {draw_names[j]}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("T [°C]")
    ax.set_title("Boiler and draw temperatures")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _time_axis(result: dict, time_unit: str) -> tuple[np.ndarray, str]:

    t_s = np.asarray(result["time"], dtype=float)
    if time_unit == "h":
        return t_s / 3600.0, "Time [h]"
    if time_unit == "s":
        return t_s, "Time [s]"
    raise ValueError("time_unit must be 'h' or 's'")


def _time_index_slice(n: int, max_points: int | None) -> slice:
    """Uniform stride so interactive HTML stays small for long runs."""
    if max_points is None or n <= max_points:
        return slice(None)
    step = int(np.ceil(n / max_points))
    return slice(None, None, step)


def plot_pipe_flow_timeseries(
    net: DHWNetwork,
    result: dict,
    *,
    time_unit: str = "h",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot stacked flow contributions for each pipe.

    Pipe subplots are stacked-area compositions built from
    ``result["pipe_contributions"]`` and show how each component
    (recirculation + fixture draws) contributes to each pipe flow.

    Parameters
    ----------
    net :
        Network used to build ``result``.
    result :
        Return value of :func:`solve_flow_rates` (keys ``time``, ``pipe_flows``,
        ``pipe_contributions``).
    time_unit :
        ``"h"`` for hours or ``"s"`` for seconds on the horizontal axis.
    figsize :
        Optional ``(width, height)`` in inches. Default scales with subplot count.
    save_path :
        If set, figure is written to this path (parent dirs are created).
    show :
        If true, call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_plot, xlabel = _time_axis(result, time_unit)

    pipe_flows = np.asarray(result["pipe_flows"])
    pipe_contributions = np.asarray(result["pipe_contributions"])
    contribution_labels = list(result["pipe_contribution_labels"])

    n_pipes = pipe_flows.shape[1]
    n_axes = n_pipes

    w, h = figsize if figsize is not None else (12.0, max(5.0, 2.0 * n_axes))
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(w, h))
    if n_axes == 1:
        axes = np.array([axes])

    names = list(net.pipe_names)
    rpi = getattr(net, "return_pipe_idx", None)
    for i in range(n_pipes):
        ax = axes[i]
        label = names[i] if i < len(names) else f"pipe {i}"
        y_stack = [pipe_contributions[:, i, k] for k in range(pipe_contributions.shape[2])]
        ax.stackplot(t_plot, *y_stack, labels=contribution_labels, alpha=0.9)
        ax.plot(t_plot, pipe_flows[:, i], color="k", linewidth=0.9, alpha=0.75)
        ax.set_ylabel("ṁ [kg/s]")
        title = f"Pipe: {label} (stacked contributions)"
        if rpi is not None and i == rpi:
            title += f" — recirc pump (ṁ fixed = {net.mdot_recirc:g} kg/s)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Keep legend compact and only show it once.
        if i == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1].set_xlabel(xlabel)

    fig.align_ylabels(axes)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_draw_timeseries(
    net: DHWNetwork,
    result: dict,
    *,
    time_unit: str = "h",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot per-draw rates and boiler cold-makeup in a separate figure."""
    t_plot, xlabel = _time_axis(result, time_unit)
    draw_rates = np.asarray(result["draw_rates"])
    mdot_cold = np.asarray(result["mdot_cold"])

    n_draws = draw_rates.shape[1]
    n_axes = n_draws + 1
    w, h = figsize if figsize is not None else (12.0, max(5.0, 2.0 * n_axes))
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(w, h))
    if n_axes == 1:
        axes = np.array([axes])

    for j in range(n_draws):
        ax = axes[j]
        dn = net.draw_nodes[j]
        ax.plot(t_plot, draw_rates[:, j], color=f"C{j + 1}", linewidth=1.0)
        ax.set_ylabel("ṁ [kg/s]")
        ax.set_title(f"Draw: {dn.name}")
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(t_plot, mdot_cold, color="C2", linewidth=1.0)
    ax.set_ylabel("ṁ [kg/s]")
    ax.set_title("Node: boiler cold makeup (Σ draws)")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)

    fig.align_ylabels(axes)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_heater_power_timeseries(
    result: dict,
    *,
    time_unit: str = "h",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot heater input and delivered thermal power over time."""
    t_plot, xlabel = _time_axis(result, time_unit)

    p_input = np.asarray(result.get("heater_power_input_W"), dtype=float)
    p_water = np.asarray(result.get("heater_power_to_water_W"), dtype=float)
    if p_input.size == 0:
        raise KeyError(
            "result must include 'heater_power_input_W'. "
            "Use FlowSolver.simulate_temperatures output."
        )
    if p_water.size == 0:
        p_water = p_input

    w, h = figsize if figsize is not None else (12.0, 4.5)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))

    ax.plot(t_plot, p_input, color="C3", linewidth=1.2, label="Heater input (electric)")
    ax.plot(t_plot, p_water, color="C1", linewidth=1.2, linestyle="--", label="Heat to water")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power [W]")
    ax.set_title("Heater power draw")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def write_timeseries_interactive_html(
    net: DHWNetwork,
    *,
    temp_result: dict,
    flow_result: dict,
    path: str | Path,
    time_unit: str = "h",
    title: str = "DHW simulation — interactive timeseries",
    max_points: int | None = 12_000,
) -> Path:
    """Write a single HTML page with Plotly timeseries (zoom, pan, box select).

    Mirrors the static matplotlib timeseries figures: temperatures, heater
    power, per-draw mass flows and cold makeup, and stacked pipe-flow
    contributions per pipe.

    Parameters
    ----------
    net :
        Network (draw names, pipe names, return pipe index for titles).
    temp_result :
        Output of :meth:`FlowSolver.simulate_temperatures`.
    flow_result :
        Output of :meth:`FlowSolver.solve_flow_rates`.
    path :
        Output ``.html`` path (parent directories are created).
    time_unit :
        ``"h"`` or ``"s"`` for the time axis.
    title :
        Browser tab / page title.
    max_points :
        If the time grid has more samples than this, data are uniformly
        strided before plotting so the HTML stays a manageable size. Use
        ``None`` to plot every timestep (can produce very large files).

    Returns
    -------
    pathlib.Path
        Path to the written HTML file.

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "Interactive HTML export requires plotly. "
            "Install project dependencies (e.g. pip install -e .)."
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_full, xlabel = _time_axis(temp_result, time_unit)
    n_t = int(np.asarray(t_full).shape[0])
    sl = _time_index_slice(n_t, max_points)
    t_plot = np.asarray(t_full, dtype=float)[sl]

    n_draws = len(net.draw_nodes)
    n_pipes = len(net.pipes)
    subplot_titles: list[str] = [
        "Boiler and draw temperatures",
        "Heater power",
        *[f"Draw: {dn.name}" for dn in net.draw_nodes],
        "Boiler cold makeup (Σ draws)",
    ]
    pipe_names = list(getattr(net, "pipe_names", [f"pipe {i}" for i in range(n_pipes)]))
    for i in range(n_pipes):
        label = pipe_names[i] if i < len(pipe_names) else f"pipe {i}"
        subplot_titles.append(f"Pipe: {label} (stacked contributions)")

    n_rows = 2 + n_draws + 1 + n_pipes
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    row = 1
    T_boiler = np.asarray(temp_result["T_boiler"], dtype=float)[sl]
    T_draw = np.asarray(temp_result["T_draw"], dtype=float)[sl, :]
    draw_names = list(temp_result["draw_names"])

    fig.add_trace(
        go.Scatter(x=t_plot, y=T_boiler, mode="lines", name="Boiler", line=dict(width=1.5)),
        row=row,
        col=1,
    )
    for j in range(T_draw.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=T_draw[:, j],
                mode="lines",
                name=f"Draw: {draw_names[j]}",
                line=dict(width=1.2),
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="T [°C]", row=row, col=1)
    row += 1

    raw_pin = temp_result.get("heater_power_input_W")
    raw_pw = temp_result.get("heater_power_to_water_W")
    p_in = np.asarray(raw_pin, dtype=float)[sl] if raw_pin is not None else np.array([])
    p_w = np.asarray(raw_pw, dtype=float)[sl] if raw_pw is not None else np.array([])
    if p_in.size:
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=p_in,
                mode="lines",
                name="Heater input (electric)",
                line=dict(width=1.2, color="#d62728"),
            ),
            row=row,
            col=1,
        )
    if p_w.size:
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=p_w,
                mode="lines",
                name="Heat to water",
                line=dict(width=1.2, color="#ff7f0e", dash="dash"),
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="Power [W]", row=row, col=1)
    row += 1

    draw_rates = np.asarray(flow_result["draw_rates"], dtype=float)[sl, :]
    for j in range(n_draws):
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=draw_rates[:, j],
                mode="lines",
                name=f"ṁ {net.draw_nodes[j].name}",
                showlegend=False,
                line=dict(width=1.0),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="ṁ [kg/s]", row=row, col=1)
        row += 1

    mdot_cold = np.asarray(flow_result["mdot_cold"], dtype=float)[sl]
    fig.add_trace(
        go.Scatter(
            x=t_plot,
            y=mdot_cold,
            mode="lines",
            name="Cold makeup",
            showlegend=False,
            line=dict(width=1.0, color="#2ca02c"),
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="ṁ [kg/s]", row=row, col=1)
    row += 1

    pipe_contributions = np.asarray(flow_result["pipe_contributions"], dtype=float)[sl, :, :]
    contribution_labels = list(flow_result["pipe_contribution_labels"])
    pipe_flows = np.asarray(flow_result["pipe_flows"], dtype=float)[sl, :]
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )

    for i in range(n_pipes):
        r = row
        for k, lab in enumerate(contribution_labels):
            color = palette[k % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=pipe_contributions[:, i, k],
                    mode="lines",
                    name=lab,
                    legendgroup=lab,
                    showlegend=(i == 0),
                    stackgroup=f"pipe_{i}",
                    fillcolor=color,
                    line=dict(width=0, color=color),
                    hovertemplate=f"{lab}<br>%{{y:.4g}} kg/s<extra></extra>",
                ),
                row=r,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=pipe_flows[:, i],
                mode="lines",
                name="Total ṁ",
                legendgroup="total",
                showlegend=(i == 0),
                line=dict(width=1.5, color="black"),
                hovertemplate="Total %{y:.4g} kg/s<extra></extra>",
            ),
            row=r,
            col=1,
        )
        fig.update_yaxes(title_text="ṁ [kg/s]", row=r, col=1)
        row += 1

    fig.update_xaxes(title_text=xlabel, row=n_rows, col=1)
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=min(3200, 140 + 200 * n_rows),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return path


def plot_pipe_temperature_heatmaps(
    net: DHWNetwork,
    result: dict,
    *,
    time_unit: str = "h",
    cmap: str = "plasma",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Heatmap of temperature along every pipe vs time.

    Each subplot shows one pipe as a 2-D image:

    * **X axis** — simulation time.
    * **Y axis** — distance from the pipe inlet [m], with cell 0
      (upstream end / boiler outlet) at the bottom.
    * **Colour** — water temperature [°C].

    Draw-node positions are marked with a dashed horizontal line and
    labelled on the right-hand side.

    Parameters
    ----------
    net :
        Network whose pipe names and draw nodes are used for labels.
    result :
        Return value of :meth:`FlowSolver.simulate_temperatures`
        (must contain ``"time"``, ``"T_pipe"``, ``"draw_names"``).
    time_unit :
        ``"h"`` (hours) or ``"s"`` (seconds) on the time axis.
    cmap :
        Matplotlib colormap name. ``"plasma"`` works well for temperature.
    figsize :
        Optional ``(width, height)`` in inches.
    save_path :
        If set, figure is saved here (parent dirs are created).
    show :
        If ``True``, call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_s = np.asarray(result["time"], dtype=float)
    t_plot, xlabel = (t_s / 3600.0, "Time [h]") if time_unit == "h" else (t_s, "Time [s]")

    T_pipe: dict[int, np.ndarray] = result["T_pipe"]   # {pipe_idx: (N_t, N_cells)}
    n_pipes = len(T_pipe)
    pipe_names = list(getattr(net, "pipe_names", [f"pipe {i}" for i in range(n_pipes)]))

    # Build a lookup of draw nodes keyed by pipe index
    draw_by_pipe: dict[int, list] = {}
    for dn in net.draw_nodes:
        draw_by_pipe.setdefault(dn.pipe_index, []).append(dn)

    w, h = figsize if figsize is not None else (14.0, max(4.0, 3.0 * n_pipes))
    fig, axes = plt.subplots(n_pipes, 1, figsize=(w, h))
    if n_pipes == 1:
        axes = np.array([axes])

    # Shared colour scale across all pipes for easy cross-pipe comparison
    all_T = np.concatenate([T_pipe[i].ravel() for i in range(n_pipes)])
    vmin, vmax = float(np.nanmin(all_T)), float(np.nanmax(all_T))

    for i, ax in enumerate(axes):
        pipe  = net.pipes[i]
        T_mat = T_pipe[i]           # (N_t, N_cells)
        N     = pipe.N_cells
        dz    = pipe.dz

        # imshow: rows → spatial (N_cells), cols → time (N_t)
        # extent = [t_left, t_right, z_bottom, z_top] with origin='upper'
        # → row 0 (inlet) appears at the top, row N-1 (outlet) at the bottom
        im = ax.imshow(
            T_mat.T,
            aspect="auto",
            origin="upper",
            extent=[t_plot[0], t_plot[-1], pipe.length, 0.0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Mark draw-node positions
        for dn in draw_by_pipe.get(i, []):
            z_dn = (dn.cell_index + 0.5) * dz   # cell-centre position [m]
            ax.axhline(z_dn, color="white", linewidth=0.9, linestyle="--", alpha=0.8)
            ax.text(
                t_plot[-1], z_dn, f"  {dn.name}",
                color="white", fontsize=7, va="center", ha="left",
                clip_on=True,
            )

        label = pipe_names[i] if i < len(pipe_names) else f"pipe {i}"
        ax.set_title(f"Pipe: {label}")
        ax.set_ylabel("Distance from inlet [m]")

        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("T [°C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = build_residential_network(dz=0.5)
    _print_network_diagram(net)
    print(f"pipe parent map: {net.parent_map}")

    t_start, t_end, dt = 0.0, 24.0 * 3600.0, 1.0
    dhw_solver = FlowSolver(net, t_start, t_end, dt)

    flow_result = dhw_solver.solve_flow_rates()
    temp_result = dhw_solver.simulate_temperatures(T_init=60.0)

    fig_ts = plot_temperature_timeseries(
        temp_result,
        time_unit="h",
        save_path="output/plots/temperature_timeseries.png",
        show=False,
    )
    plt.close(fig_ts)

    fig_hm = plot_pipe_temperature_heatmaps(
        net,
        temp_result,
        time_unit="h",
        save_path="output/plots/temperature_heatmaps.png",
        show=False,
    )
    plt.close(fig_hm)

    fig_pf = plot_pipe_flow_timeseries(
        net,
        flow_result,
        time_unit="h",
        save_path="output/plots/pipe_flow_timeseries.png",
        show=False,
    )
    plt.close(fig_pf)

    fig_dr = plot_draw_timeseries(
        net,
        flow_result,
        time_unit="h",
        save_path="output/plots/draw_timeseries.png",
        show=False,
    )
    plt.close(fig_dr)

    fig_hp = plot_heater_power_timeseries(
        temp_result,
        time_unit="h",
        save_path="output/plots/heater_power_timeseries.png",
        show=False,
    )
    plt.close(fig_hp)

    write_timeseries_interactive_html(
        net,
        temp_result=temp_result,
        flow_result=flow_result,
        path="output/plots/timeseries_interactive.html",
        time_unit="h",
    )

    heater_energy = dhw_solver.compute_heater_energy(temp_result)
    print(heater_energy)
