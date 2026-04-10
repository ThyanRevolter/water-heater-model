"""
DHW Recirculation Network: Thermal-Hydraulic Model
===================================================

Implicit (backward Euler) finite-difference solver for a domestic hot water
recirculation loop with draw-off nodes.

Governing equation per pipe segment i (interior node):

    c * m_i * dT_i/dt = q_i(t)
                        - h * S_i * (T_i - T_s)
                        - mdot(t) * c * (T_i - T_{i-1})
                        + (k_eff * A_c / dz) * (T_{i+1} - 2*T_i + T_{i-1})

This is the energy balance from Savatorova & Talonov (SIMIODE EXPO 2022),
extended to a pipe network with multiple draw-off nodes following the
advection model of Moss & Critoph (Energy & Buildings, 2022) and the
circulation-loss framework of Klimczak et al. (Energies, 2022).

The network topology is defined via a pype_schema-compatible JSON file.

References
----------
[1] Savatorova, V. & Talonov, A. (2022). Teaching differential equations
    through modeling: hot water heater. SIMIODE EXPO 2022.
[2] Moss, R.W. & Critoph, R.E. (2022). Optimisation of a recirculating
    domestic hot water system to minimise wait time and heat loss.
    Energy & Buildings, 260, 111850.
[3] Klimczak, M., Bartnicki, G. & Ziembicki, P. (2022). Energy Consumption
    by DHW System with a Circulation Loop. Energies, 15, 3952.
[4] Cengel, Y.A. & Cimbala, J.M. (2018). Fluid Mechanics: Fundamentals
    and Applications, 4th ed. McGraw-Hill.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


# ──────────────────────────────────────────────────────────────
# 1. DATA CLASSES  (network-level parameters)
# ──────────────────────────────────────────────────────────────

@dataclass
class PipeParams:
    """Physical parameters for a single pipe segment."""
    length: float          # m
    inner_diameter: float  # m
    wall_thickness: float  # m
    insulation_k: float    # W/(m·K)  insulation thermal conductivity
    insulation_t: float    # m        insulation thickness
    h_ext: float           # W/(m²·K) external convection coefficient
    pipe_k: float          # W/(m·K)  pipe wall conductivity (copper ~385)
    N_cells: int           # number of finite-volume cells along pipe


@dataclass
class DrawNode:
    """A point where water can be drawn from the network."""
    name: str
    pipe_index: int                # index into the pipes list
    cell_index: int                # cell along that pipe
    draw_profile: Callable         # f(t) -> mass flow rate [kg/s]


@dataclass
class DHWNetwork:
    """Complete DHW recirculation network definition."""
    # Physical constants
    c: float = 4186.0              # J/(kg·K)  specific heat of water
    rho: float = 998.0             # kg/m³     water density
    T_supply: float = 60.0         # °C        boiler setpoint
    T_cold: float = 15.0           # °C        cold-water makeup temperature
    T_ambient: float = 20.0        # °C        surroundings
    heater_power: float = 4500.0   # W         electric element
    boiler_volume: float = 0.189   # m³        ~50 gal tank
    mdot_recirc: float = 0.006     # kg/s      recirculation pump flow

    pipes: List[PipeParams] = field(default_factory=list)
    draw_nodes: List[DrawNode] = field(default_factory=list)

    # Adjacency: pipes[i] feeds into pipes[j] (list of (i, j, cell_out, cell_in))
    adjacency: List[tuple] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# 2. BUILD A SAMPLE NETWORK (matches the JSON topology)
# ──────────────────────────────────────────────────────────────

def default_pipe(length=5.0, N=10, d_inner=0.02, insul_t=0.025):
    """Return a PipeParams with reasonable residential defaults."""
    return PipeParams(
        length=length,
        inner_diameter=d_inner,
        wall_thickness=0.001,       # 1 mm copper
        insulation_k=0.035,         # W/(m·K) foam insulation
        insulation_t=insul_t,
        h_ext=10.0,                 # W/(m²·K) free convection to air
        pipe_k=385.0,               # copper
        N_cells=N,
    )


def build_sample_network() -> DHWNetwork:
    """
    Build the network corresponding to dhw_network.json:

      Boiler -> supply_main -> junction_F1 ─┬─> bath1_shower  (draw)
                                             ├─> bath1_faucet  (draw)
                                             └─> riser_F2 -> junction_F2
                                                   ├─> bath2_shower  (draw)
                                                   ├─> bath2_faucet  (draw)
                                                   └─> kitchen_sink  (draw)
                                                         └─> recirc_return -> Boiler
    """
    net = DHWNetwork()

    # Pipe indices (0-based)
    #  0: supply_main         Boiler → Junction_F1        8 m
    #  1: branch_b1_shower    Junction_F1 → Bath1 shower  3 m
    #  2: branch_b1_faucet    Junction_F1 → Bath1 faucet  2 m
    #  3: riser_F2            Junction_F1 → Junction_F2   4 m
    #  4: branch_b2_shower    Junction_F2 → Bath2 shower  3 m
    #  5: branch_b2_faucet    Junction_F2 → Bath2 faucet  2 m
    #  6: branch_kitchen      Junction_F2 → Kitchen sink  5 m
    #  7: recirc_return       Kitchen end → Boiler         12 m (return)

    N = 8  # cells per pipe (for manageability)
    net.pipes = [
        default_pipe(length=8.0,  N=N),   # 0  supply main
        default_pipe(length=3.0,  N=N),   # 1  bath1 shower
        default_pipe(length=2.0,  N=N),   # 2  bath1 faucet
        default_pipe(length=4.0,  N=N),   # 3  riser to floor 2
        default_pipe(length=3.0,  N=N),   # 4  bath2 shower
        default_pipe(length=2.0,  N=N),   # 5  bath2 faucet
        default_pipe(length=5.0,  N=N),   # 6  kitchen sink
        default_pipe(length=12.0, N=N, d_inner=0.012, insul_t=0.02),  # 7 return (smaller bore)
    ]

    # Adjacency: (from_pipe, to_pipe)
    # The last cell of from_pipe feeds the first cell of to_pipe
    net.adjacency = [
        (0, 1),   # supply → bath1_shower branch
        (0, 2),   # supply → bath1_faucet branch
        (0, 3),   # supply → riser_F2
        (3, 4),   # riser  → bath2_shower
        (3, 5),   # riser  → bath2_faucet
        (3, 6),   # riser  → kitchen
        (6, 7),   # kitchen end → recirc return
        # (7, boiler) handled as BC
    ]

    # ── Draw-off profiles (step functions with stochastic peaks) ──
    def make_draw_profile(peak_kg_s, on_hours):
        """Return a function f(t_seconds) -> mdot [kg/s]."""
        def profile(t):
            hour = (t / 3600.0) % 24.0
            for (h_start, h_end) in on_hours:
                if h_start <= hour < h_end:
                    return peak_kg_s
            return 0.0
        return profile

    #                              name               pipe  cell  profile
    net.draw_nodes = [
        DrawNode("Bathroom1_Shower",  1, N-1,
                 make_draw_profile(0.15, [(7.0, 7.25), (22.0, 22.15)])),
        DrawNode("Bathroom1_Faucet",  2, N-1,
                 make_draw_profile(0.05, [(7.25, 7.33), (12.0, 12.08)])),
        DrawNode("Bathroom2_Shower",  4, N-1,
                 make_draw_profile(0.15, [(7.5, 7.75)])),
        DrawNode("Bathroom2_Faucet",  5, N-1,
                 make_draw_profile(0.03, [(8.0, 8.08), (18.0, 18.08)])),
        DrawNode("Kitchen_Sink",      6, N-1,
                 make_draw_profile(0.10, [(7.0, 7.08), (12.0, 12.17),
                                          (18.0, 18.33)])),
    ]

    return net


# ──────────────────────────────────────────────────────────────
# 3. IMPLICIT SOLVER (backward Euler)
# ──────────────────────────────────────────────────────────────

def compute_pipe_UA(p: PipeParams):
    """
    Overall heat loss coefficient per unit length [W/(m·K)].
    Uses the radial conduction + external convection model from
    Klimczak et al. (2022), Eq. (2).

    U_linear = pi / [ 1/(h_in*d_i) + ln(d_o/d_i)/(2*k_pipe)
                     + ln(d_ins/d_o)/(2*k_ins) + 1/(h_ext*d_ins) ]

    For simplicity h_in → ∞ (turbulent water), so first term ≈ 0.
    """
    d_i = p.inner_diameter
    d_o = d_i + 2 * p.wall_thickness
    d_ins = d_o + 2 * p.insulation_t

    R_pipe = np.log(d_o / d_i) / (2 * np.pi * p.pipe_k)
    R_ins  = np.log(d_ins / d_o) / (2 * np.pi * p.insulation_k)
    R_ext  = 1.0 / (np.pi * d_ins * p.h_ext)

    U_per_m = 1.0 / (R_pipe + R_ins + R_ext)   # W/(m·K)
    return U_per_m


def build_global_system(net: DHWNetwork, T_old: np.ndarray, t: float, dt: float):
    """
    Assemble the global implicit system  A·T_new = b  using backward Euler.

    For each pipe cell i the energy balance (semi-discrete) is:

        rho*A_c*dz*c * (T_i^{n+1} - T_i^n)/dt
            = - U_L * dz * (T_i^{n+1} - T_amb)                      [wall loss]
              - mdot_net * c * (T_i^{n+1} - T_{i-1}^{n+1})          [advection, upwind]
              + k_eff*A_c/dz * (T_{i+1}^{n+1} - 2*T_i^{n+1} + T_{i-1}^{n+1}) [diffusion]

    Rearranged into implicit form:  a_W T_{i-1} + a_P T_i + a_E T_{i+1} = S

    Boiler node (separate lumped ODE, also implicit):
        c*m_boiler*(T_b^{n+1}-T_b^n)/dt = q_heater - h*S_b*(T_b^{n+1}-T_amb)
                                            - mdot_out*c*(T_b^{n+1} - T_return)
                                            + mdot_cold*c*(T_cold - T_b^{n+1})
    """
    n_pipes = len(net.pipes)
    cells_per_pipe = [p.N_cells for p in net.pipes]
    N_pipe_cells = sum(cells_per_pipe)
    N_total = N_pipe_cells + 1   # +1 for boiler lumped node

    # Global indexing
    offsets = np.zeros(n_pipes + 1, dtype=int)
    for k in range(n_pipes):
        offsets[k + 1] = offsets[k] + cells_per_pipe[k]
    boiler_idx = N_total - 1

    A_rows, A_cols, A_vals = [], [], []
    b = np.zeros(N_total)

    def add_entry(i, j, v):
        A_rows.append(i)
        A_cols.append(j)
        A_vals.append(v)

    # ── Compute draw-offs at this time ──
    draw_at_pipe_cell = {}  # (pipe_idx, cell_idx) -> total draw [kg/s]
    for dn in net.draw_nodes:
        key = (dn.pipe_index, dn.cell_index)
        draw_at_pipe_cell[key] = draw_at_pipe_cell.get(key, 0.0) + dn.draw_profile(t)

    total_draw = sum(draw_at_pipe_cell.values())
    mdot_cold_makeup = total_draw  # mass balance: cold water replaces drawn water

    # ── Total flow through each pipe ──
    # Simple hydraulic assumption: flow in each pipe = recirc + downstream draws
    # (we walk from terminal pipes backward)
    pipe_flow = np.zeros(n_pipes)  # kg/s
    # Build downstream draw for each pipe
    # Pipe 7 (return): only recirc flow
    pipe_flow[7] = net.mdot_recirc
    # Pipe 6 (kitchen): recirc + kitchen draw
    pipe_flow[6] = net.mdot_recirc + draw_at_pipe_cell.get((6, cells_per_pipe[6]-1), 0.0)
    # Pipes 4,5 (bath2): only their own draw
    pipe_flow[4] = draw_at_pipe_cell.get((4, cells_per_pipe[4]-1), 0.0)
    pipe_flow[5] = draw_at_pipe_cell.get((5, cells_per_pipe[5]-1), 0.0)
    # Pipe 3 (riser F2): feeds pipes 4,5,6
    pipe_flow[3] = pipe_flow[4] + pipe_flow[5] + pipe_flow[6]
    # Pipes 1,2 (bath1): own draw only
    pipe_flow[1] = draw_at_pipe_cell.get((1, cells_per_pipe[1]-1), 0.0)
    pipe_flow[2] = draw_at_pipe_cell.get((2, cells_per_pipe[2]-1), 0.0)
    # Pipe 0 (supply main): feeds all downstream
    pipe_flow[0] = pipe_flow[1] + pipe_flow[2] + pipe_flow[3]

    # effective axial conductivity (static + turbulent mixing)
    k_eff = 0.60 + 0.05  # W/(m·K)  water k ≈ 0.60, small turbulent augmentation

    # ── Assemble pipe cells ──
    for p_idx, pipe in enumerate(net.pipes):
        N_c = pipe.N_cells
        dz = pipe.length / N_c
        A_c = np.pi / 4 * pipe.inner_diameter ** 2
        m_cell = net.rho * A_c * dz           # kg per cell
        U_L = compute_pipe_UA(pipe)            # W/(m·K)
        mdot = pipe_flow[p_idx]                # kg/s through this pipe

        # Coefficients (backward Euler)
        alpha = m_cell * net.c / dt
        beta  = U_L * dz                       # wall loss coeff
        gamma = mdot * net.c                    # advection coeff (upwind)
        kappa = k_eff * A_c / dz               # diffusion coeff

        for j in range(N_c):
            gi = offsets[p_idx] + j  # global index

            # Diagonal (accumulation + loss + advection_out + 2*diffusion)
            a_P = alpha + beta + gamma + 2 * kappa

            # Source
            S = alpha * T_old[gi] + beta * net.T_ambient

            # Draw-off at this cell (extra advection out, replaced by cold water)
            mdot_draw = draw_at_pipe_cell.get((p_idx, j), 0.0)
            if mdot_draw > 0:
                a_P += mdot_draw * net.c
                S += mdot_draw * net.c * net.T_cold

            # West neighbor (advection in + diffusion)
            if j > 0:
                add_entry(gi, gi - 1, -(gamma + kappa))
            else:
                # First cell: upstream is either boiler or another pipe
                # Identify upstream source
                if p_idx == 0:
                    # upstream = boiler node
                    add_entry(gi, boiler_idx, -(gamma + kappa))
                elif p_idx == 7:
                    # return pipe, upstream = last cell of pipe 6
                    upstream_gi = offsets[6] + cells_per_pipe[6] - 1
                    add_entry(gi, upstream_gi, -(gamma + kappa))
                else:
                    # branches: upstream = last cell of parent pipe
                    parent_map = {1: 0, 2: 0, 3: 0, 4: 3, 5: 3, 6: 3}
                    parent_pipe = parent_map.get(p_idx, 0)
                    upstream_gi = offsets[parent_pipe] + cells_per_pipe[parent_pipe] - 1
                    add_entry(gi, upstream_gi, -(gamma + kappa))

            # East neighbor (diffusion only, upwind scheme puts advection on west)
            if j < N_c - 1:
                add_entry(gi, gi + 1, -kappa)
            else:
                # Last cell: adiabatic-ish BC (extrapolate) → just remove east diffusion
                a_P -= kappa  # compensate for missing east neighbor

            add_entry(gi, gi, a_P)
            b[gi] = S

    # ── Boiler lumped node ──
    m_boiler = net.rho * net.boiler_volume
    T_b_old = T_old[boiler_idx]

    # Boiler surface area estimate (cylinder h ≈ 1.2 m, d ≈ 0.5 m)
    S_boiler = np.pi * 0.5 * 1.2 + 2 * np.pi * 0.25**2
    h_boiler = 5.0  # W/(m²·K)  well-insulated tank

    # Thermostat: heater ON if T_b < T_supply, OFF if T_b >= T_supply + 2
    if T_b_old < net.T_supply:
        q_heat = net.heater_power
    elif T_b_old >= net.T_supply + 2.0:
        q_heat = 0.0
    else:
        q_heat = net.heater_power  # hysteresis band

    alpha_b = m_boiler * net.c / dt
    beta_b  = h_boiler * S_boiler
    mdot_out = pipe_flow[0]          # total flow leaving boiler into pipe 0
    mdot_return = net.mdot_recirc    # from recirc return pipe

    a_P_b = alpha_b + beta_b + mdot_out * net.c + mdot_cold_makeup * net.c
    S_b = (alpha_b * T_b_old
           + beta_b * net.T_ambient
           + q_heat
           + mdot_cold_makeup * net.c * net.T_cold)

    # Return water enters boiler (implicit coupling to last cell of pipe 7)
    return_gi = offsets[7] + cells_per_pipe[7] - 1
    add_entry(boiler_idx, return_gi, -mdot_return * net.c)
    a_P_b += mdot_return * net.c  # moved to LHS with sign

    add_entry(boiler_idx, boiler_idx, a_P_b)
    b[boiler_idx] = S_b

    # ── Build sparse matrix ──
    from scipy.sparse import coo_matrix
    A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(N_total, N_total)).tocsr()

    return A, b, boiler_idx, offsets, pipe_flow


def simulate(net: DHWNetwork, t_end: float = 86400.0, dt: float = 10.0):
    """
    Run the implicit transient simulation.

    Parameters
    ----------
    net : DHWNetwork
    t_end : float   total simulation time [s] (default 24 h)
    dt : float      time step [s]

    Returns
    -------
    times : 1-D array
    T_history : 2-D array  (n_steps, N_total)
    metadata : dict with indexing info
    """
    n_pipes = len(net.pipes)
    cells_per_pipe = [p.N_cells for p in net.pipes]
    N_pipe_cells = sum(cells_per_pipe)
    N_total = N_pipe_cells + 1

    # Initial condition: everything at ambient except boiler at setpoint
    T = np.full(N_total, net.T_ambient)
    T[-1] = net.T_supply  # boiler starts hot

    n_steps = int(np.ceil(t_end / dt))
    save_every = max(1, n_steps // 2000)  # keep ≤ 2000 snapshots
    times = []
    T_history = []

    for step in range(n_steps):
        t = step * dt
        A, b, boiler_idx, offsets, _ = build_global_system(net, T, t, dt)
        T_new = spsolve(A, b)
        T = T_new.copy()

        if step % save_every == 0:
            times.append(t)
            T_history.append(T.copy())

    times = np.array(times)
    T_history = np.array(T_history)

    metadata = {
        "offsets": offsets,
        "boiler_idx": boiler_idx,
        "cells_per_pipe": cells_per_pipe,
        "pipe_names": [
            "supply_main", "bath1_shower", "bath1_faucet",
            "riser_F2", "bath2_shower", "bath2_faucet",
            "kitchen", "recirc_return"
        ],
        "draw_node_names": [dn.name for dn in net.draw_nodes],
        "draw_node_global_idx": [
            offsets[dn.pipe_index] + dn.cell_index for dn in net.draw_nodes
        ],
    }

    return times, T_history, metadata


# ──────────────────────────────────────────────────────────────
# 4. POST-PROCESSING & PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_results(times, T_history, metadata, save_path="dhw_results.png"):
    hours = times / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (a) Boiler temperature
    ax = axes[0]
    ax.plot(hours, T_history[:, metadata["boiler_idx"]], "r-", linewidth=1.5)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("(a) Boiler Tank Temperature")
    ax.axhline(60, color="gray", linestyle="--", linewidth=0.8, label="Setpoint 60 °C")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Draw-node delivery temperatures
    ax = axes[1]
    for name, gi in zip(metadata["draw_node_names"],
                        metadata["draw_node_global_idx"]):
        ax.plot(hours, T_history[:, gi], linewidth=1.2, label=name)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("(b) Water Temperature at Draw-off Points")
    ax.axhline(40, color="gray", linestyle="--", linewidth=0.8, label="Min comfort 40 °C")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) Recirculation return temperature (last cell of pipe 7)
    ax = axes[2]
    offsets = metadata["offsets"]
    ret_idx = offsets[7] + metadata["cells_per_pipe"][7] - 1
    ax.plot(hours, T_history[:, ret_idx], "b-", linewidth=1.5, label="Recirc return")
    ax.plot(hours, T_history[:, metadata["boiler_idx"]], "r--",
            linewidth=1.0, alpha=0.6, label="Boiler")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("(c) Recirculation Return vs Boiler Temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def compute_energy_metrics(times, T_history, metadata, net):
    """Compute and print energy performance metrics."""
    dt_arr = np.diff(times)
    T_boiler = T_history[:, metadata["boiler_idx"]]

    # Estimate heater ON time (when boiler T < setpoint)
    heater_on = T_boiler[:-1] < net.T_supply
    energy_heater_J = np.sum(heater_on * net.heater_power * dt_arr)
    energy_heater_kWh = energy_heater_J / 3.6e6

    # Estimate total wall heat loss (all pipes)
    total_loss_W = []
    for step_idx in range(len(times)):
        loss = 0.0
        for p_idx, pipe in enumerate(net.pipes):
            off = metadata["offsets"][p_idx]
            N_c = pipe.N_cells
            dz = pipe.length / N_c
            U_L = compute_pipe_UA(pipe)
            for j in range(N_c):
                loss += U_L * dz * (T_history[step_idx, off + j] - net.T_ambient)
        total_loss_W.append(loss)
    total_loss_W = np.array(total_loss_W)
    avg_loss_W = np.mean(total_loss_W)

    print("\n" + "="*55)
    print("  DHW RECIRCULATION SYSTEM — ENERGY SUMMARY")
    print("="*55)
    print(f"  Simulation period     : {times[-1]/3600:.1f} hours")
    print(f"  Heater energy input   : {energy_heater_kWh:.2f} kWh")
    print(f"  Avg pipe heat loss    : {avg_loss_W:.1f} W")
    print(f"  Recirc pump flow      : {net.mdot_recirc*1000:.1f} g/s")
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building DHW recirculation network...")
    net = build_sample_network()

    print(f"  {len(net.pipes)} pipe segments, "
          f"{sum(p.N_cells for p in net.pipes)} cells + 1 boiler node")
    print(f"  {len(net.draw_nodes)} draw-off points")
    print(f"  Recirculation flow: {net.mdot_recirc*1000:.1f} g/s")

    # Load and echo the pype_schema JSON for reference
    try:
        with open("dhw_network.json") as f:
            schema = json.load(f)
        print(f"\n  pype_schema nodes: {schema['nodes']}")
        facility_nodes = schema["DHWSystem"]["nodes"]
        print(f"  Facility sub-nodes: {facility_nodes}")
    except FileNotFoundError:
        print("  (dhw_network.json not found, skipping schema echo)")

    print("\nRunning 24-hour implicit simulation (dt=10 s)...")
    times, T_hist, meta = simulate(net, t_end=86400.0, dt=10.0)
    print(f"  Done. {len(times)} snapshots saved.")

    plot_results(times, T_hist, meta, save_path="dhw_results.png")
    compute_energy_metrics(times, T_hist, meta, net)
