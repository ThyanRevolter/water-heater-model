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

from dhw import DHWNetwork, DrawNode
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
        mdot_recirc=0.3,
        T_ambient=20.0,
        T_cold=15.0,
        return_pipe_idx=3,
        pipe_flow_fn=_residential_pipe_flow,
    )

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
    shower_csv = generate_draw_profile_csv(
        profiles_dir / "shower.csv",
        peak_kg_s=0.15,
        on_windows_hours=[(7.0, 7.25), (22.0, 22.17)],
    )
    faucet_csv = generate_draw_profile_csv(
        profiles_dir / "faucet.csv",
        peak_kg_s=0.05,
        on_windows_hours=[(7.25, 7.33), (12.0, 12.08), (22.17, 22.25)],
    )
    sink_csv = generate_draw_profile_csv(
        profiles_dir / "sink.csv",
        peak_kg_s=0.10,
        on_windows_hours=[(7.0, 7.08), (12.0, 12.17), (18.0, 18.33)],
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
# Flow-rate solver over a simulation window
# ──────────────────────────────────────────────────────────────────────

def solve_flow_rates(
    net: DHWNetwork,
    t_start: float,
    t_end: float,
    dt: float = 1.0,
) -> dict:
    """Compute pipe flow rates at every timestep.

    Parameters
    ----------
    net :
        Network built by :func:`build_residential_network`.
    t_start, t_end :
        Simulation window, in seconds.
    dt :
        Timestep, in seconds. Defaults to 1 s.

    Returns
    -------
    dict
        ``time``       : 1-D array of time values ``(N_steps,)``.
        ``pipe_flows`` : 2-D array ``(N_steps, N_pipes)`` — ṁ in each pipe.
        ``mdot_cold``  : 1-D array ``(N_steps,)`` — cold-makeup mass flow.
        ``draw_rates`` : 2-D array ``(N_steps, N_draws)`` — per-draw rate.
    """
    times = np.arange(t_start, t_end, dt)
    n_steps = len(times)
    n_pipes = len(net.pipes)
    n_draws = len(net.draw_nodes)

    pipe_flows = np.zeros((n_steps, n_pipes))
    draw_rates = np.zeros((n_steps, n_draws))
    mdot_cold = np.zeros(n_steps)

    for k, t in enumerate(times):
        draw_at_pipe_cell: dict[tuple[int, int], float] = {}
        for j, dn in enumerate(net.draw_nodes):
            mdot = dn.draw_profile(t)
            draw_rates[k, j] = mdot
            draw_at_pipe_cell[(dn.pipe_index, dn.cell_index)] = mdot

        q = net.pipe_flow_fn(net, draw_at_pipe_cell)
        pipe_flows[k, :] = q
        mdot_cold[k] = draw_rates[k, :].sum()

    return {
        "time": times,
        "pipe_flows": pipe_flows,
        "mdot_cold": mdot_cold,
        "draw_rates": draw_rates,
    }


def plot_flow_timeseries(
    net: DHWNetwork,
    result: dict,
    *,
    time_unit: str = "h",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot pipe ṁ and node-side flows (draws + cold makeup) as stacked subplots.

    Pipe flows are the scalar ṁ per pipe from :func:`solve_flow_rates`. The
    recirculation pump does not appear as its own curve: in this model the pump
    fixes ṁ to ``net.mdot_recirc`` in the return pipe (see ``net.return_pipe_idx``),
    so that pipe’s subplot is the pump delivery (typically a flat line).

    Node flows are the draw rates at each :class:`~dhw.models.DrawNode` and the
    boiler cold-makeup rate (Σ draws).

    Parameters
    ----------
    net :
        Network used to build ``result``.
    result :
        Return value of :func:`solve_flow_rates` (keys ``time``, ``pipe_flows``,
        ``draw_rates``, ``mdot_cold``).
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
    t_s = np.asarray(result["time"], dtype=float)
    if time_unit == "h":
        t_plot = t_s / 3600.0
        xlabel = "Time [h]"
    elif time_unit == "s":
        t_plot = t_s
        xlabel = "Time [s]"
    else:
        raise ValueError("time_unit must be 'h' or 's'")

    pipe_flows = np.asarray(result["pipe_flows"])
    draw_rates = np.asarray(result["draw_rates"])
    mdot_cold = np.asarray(result["mdot_cold"])

    n_pipes = pipe_flows.shape[1]
    n_draws = draw_rates.shape[1]
    n_axes = n_pipes + n_draws + 1

    w, h = figsize if figsize is not None else (12.0, max(6.0, 2.0 * n_axes))
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(w, h))
    if n_axes == 1:
        axes = np.array([axes])

    names = list(net.pipe_names)
    rpi = getattr(net, "return_pipe_idx", None)
    for i in range(n_pipes):
        ax = axes[i]
        label = names[i] if i < len(names) else f"pipe {i}"
        ax.plot(t_plot, pipe_flows[:, i], color="C0", linewidth=1.0)
        ax.set_ylabel("ṁ [kg/s]")
        title = f"Pipe: {label}"
        if rpi is not None and i == rpi:
            title += f" — recirc pump (ṁ fixed = {net.mdot_recirc:g} kg/s)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    base = n_pipes
    for j in range(n_draws):
        ax = axes[base + j]
        dn = net.draw_nodes[j]
        ax.plot(t_plot, draw_rates[:, j], color="C1", linewidth=1.0)
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


# ──────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────────────────

def _print_discretisation(net: DHWNetwork, dz: float) -> None:
    print(f"  Target dz = {dz} m — pipe discretisation:")
    for name, pipe in zip(net.pipe_names, net.pipes):
        actual_dz = pipe.length / pipe.N_cells
        print(f"    {name:<18s}  L={pipe.length:5.1f} m  "
              f"N={pipe.N_cells:3d} cells  dz={actual_dz:.3f} m")


# ──────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = build_residential_network(dz=0.5)

    t_start, t_end, dt = 0.0, 24.0 * 3600.0, 1.0
    result = solve_flow_rates(net, t_start, t_end, dt)

    fig = plot_flow_timeseries(
        net,
        result,
        save_path="output/plots/flow_timeseries_pipes_nodes.png",
        show=False,
    )
    plt.close(fig)
    print("Wrote output/plots/flow_timeseries_pipes_nodes.png")

    def snapshot(tag: str, t_sample_h: float) -> None:
        idx = int((t_sample_h * 3600.0 - t_start) / dt)
        print(f"\n  Flow rates at t = {t_sample_h:.2f} h  [{tag}]:")
        for p, name in enumerate(net.pipe_names):
            print(f"    {name:<18s}  ṁ = {result['pipe_flows'][idx, p]:.4f} kg/s")
        print(f"    {'cold_makeup':<18s}  ṁ = {result['mdot_cold'][idx]:.4f} kg/s")
        print(f"  Draws at t = {t_sample_h:.2f} h:")
        for j, dn in enumerate(net.draw_nodes):
            print(f"    {dn.name:<18s}  ṁ = {result['draw_rates'][idx, j]:.4f} kg/s")

    snapshot("morning shower",       7.10)
    snapshot("mid-morning faucet",   7.30)
    snapshot("lunchtime sink+faucet", 12.05)
    snapshot("quiet pre-dawn",       3.00)
