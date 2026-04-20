"""
Implicit (backward Euler) finite-difference solver for a DHW network.

Node-type governing equations
==============================

All pipe cells share the base advection–diffusion–loss balance:

    rho*A_c*dz*c * (T_i^{n+1} - T_i^n)/dt
        = - U_L*dz * (T_i^{n+1} - T_amb)            [wall loss]
          - mdot*c * (T_i^{n+1} - T_{i-1}^{n+1})    [advection, upwind]
          + k_eff*A_c/dz * (T_{i+1} - 2*T_i + T_{i-1})^{n+1}  [axial diffusion]

Additional terms by node type
------------------------------

Draw node  (DrawNode at cell i):
    No extra cell terms.  The branch pipe_flow is set equal to the draw rate,
    so the upwind advection (gamma = mdot_draw*c already in a_P) carries hot
    water to the cell and lets it exit the domain through the east-face BC.
    Cold makeup water enters the system at the boiler tank (see boiler node).

Heat-exchanger node  (HeatExchangerNode at cell i):
    - UA_hx * (T_i^{n+1} - T_process)
    → heat extracted to a fixed-temperature process reservoir; no mass drawn.
    Implicit: a_P += UA_hx,  S += UA_hx*T_process

Boiler lumped node:
    m_b*c * (T_b^{n+1} - T_b^n)/dt
        = q_heat                                      [heating element, thermostat]
          - h*S_b * (T_b^{n+1} - T_amb)              [tank standby loss]
          - mdot_out*c * T_b^{n+1}                   [hot water leaving]
          + mdot_cold*c * T_cold                      [cold makeup, =total draw]
          + mdot_ret*c * T_ret^{n+1}                 [recirculation return, implicit]

References
----------
[1] Savatorova & Talonov (SIMIODE EXPO 2022)
[2] Moss & Critoph (Energy & Buildings, 2022)
[3] Klimczak et al. (Energies, 2022)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .models import DHWNetwork, PipeParams


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


def build_global_system(net: DHWNetwork, T_old: np.ndarray, t: float, dt: float):
    """
    Assemble the global implicit system  A·T_new = b  (backward Euler).

    The network topology is driven by:
      net.parent_map     — {pipe_idx: parent_pipe_idx} for upstream BC
      net.return_pipe_idx — which pipe's last cell feeds the boiler return
      net.pipe_flow_fn   — callable(net, draw_at_pipe_cell) -> pipe_flow array

    Returns
    -------
    A          : sparse CSR matrix  (N_total × N_total)
    b          : RHS vector         (N_total,)
    boiler_idx : global index of the boiler node
    offsets    : per-pipe global index offsets (length n_pipes+1)
    pipe_flow  : mass flow rate through each pipe [kg/s]
    """
    if net.pipe_flow_fn is None:
        raise ValueError("DHWNetwork.pipe_flow_fn must be set before solving.")

    n_pipes = len(net.pipes)
    cells_per_pipe = [p.N_cells for p in net.pipes]
    N_pipe_cells = sum(cells_per_pipe)
    N_total = N_pipe_cells + 1   # +1 for boiler lumped node

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

    # ── Draw-offs at this time step ──
    draw_at_pipe_cell = {}
    for dn in net.draw_nodes:
        key = (dn.pipe_index, dn.cell_index)
        draw_at_pipe_cell[key] = draw_at_pipe_cell.get(key, 0.0) + dn.draw_profile(t)

    total_draw = sum(draw_at_pipe_cell.values())
    mdot_cold_makeup = total_draw

    # ── HX contributions indexed by cell ──
    hx_at_pipe_cell = {}
    for hx in net.hx_nodes:
        key = (hx.pipe_index, hx.cell_index)
        hx_at_pipe_cell[key] = hx_at_pipe_cell.get(key, (0.0, 0.0))
        ua, tp = hx_at_pipe_cell[key]
        # combined UA and UA-weighted process temperature for multiple HXs on same cell
        ua_new = ua + hx.UA_hx
        tp_new = (ua * tp + hx.UA_hx * hx.T_process) / ua_new
        hx_at_pipe_cell[key] = (ua_new, tp_new)

    # ── Pipe flow rates ──
    pipe_flow = net.pipe_flow_fn(net, draw_at_pipe_cell)

    k_eff = 0.60 + 0.05  # W/(m·K)  water k + small turbulent augmentation

    # ── Assemble pipe cells ──
    for p_idx, pipe in enumerate(net.pipes):
        N_c = pipe.N_cells
        dz = pipe.length / N_c
        A_c = np.pi / 4 * pipe.inner_diameter ** 2
        m_cell = net.rho * A_c * dz
        U_L = compute_pipe_UA(pipe)
        mdot = pipe_flow[p_idx]

        alpha = m_cell * net.c / dt    # accumulation
        beta  = U_L * dz               # wall loss
        gamma = mdot * net.c           # advection (upwind)
        kappa = k_eff * A_c / dz       # axial diffusion

        for j in range(N_c):
            gi = offsets[p_idx] + j

            a_P = alpha + beta + gamma + 2 * kappa
            S   = alpha * T_old[gi] + beta * net.T_ambient

            # ── Draw node contribution ──
            # Cold water makeup enters at the boiler tank (handled in the boiler
            # node equation), not at the draw point.  The upwind advection term
            # (gamma already in a_P) already carries water through the branch and
            # out of the domain at this cell — no extra outflow term is needed.
            mdot_draw = draw_at_pipe_cell.get((p_idx, j), 0.0)

            # ── Heat exchanger node contribution ──
            if (p_idx, j) in hx_at_pipe_cell:
                ua, t_proc = hx_at_pipe_cell[(p_idx, j)]
                a_P += ua
                S   += ua * t_proc

            # West neighbor (advection in + diffusion)
            if j > 0:
                add_entry(gi, gi - 1, -(gamma + kappa))
            else:
                # First cell: upstream is boiler (pipe 0) or parent pipe's last cell
                if p_idx == 0:
                    add_entry(gi, boiler_idx, -(gamma + kappa))
                else:
                    parent_pipe = net.parent_map[p_idx]
                    upstream_gi = offsets[parent_pipe] + cells_per_pipe[parent_pipe] - 1
                    add_entry(gi, upstream_gi, -(gamma + kappa))

            # East neighbor (diffusion only; advection is upwind → west side)
            if j < N_c - 1:
                add_entry(gi, gi + 1, -kappa)
            else:
                a_P -= kappa   # adiabatic east BC: remove phantom east diffusion

            add_entry(gi, gi, a_P)
            b[gi] = S

    # ── Boiler lumped node ──
    #
    # m_b*c*(T_b^{n+1} - T_b^n)/dt
    #   = q_heat - h*S_b*(T_b^{n+1}-T_amb) - mdot_out*c*T_b^{n+1}
    #     + mdot_cold*c*T_cold + mdot_ret*c*T_ret^{n+1}
    #
    m_boiler = net.rho * net.boiler_volume
    T_b_old  = T_old[boiler_idx]

    S_boiler = np.pi * 0.5 * 1.2 + 2 * np.pi * 0.25**2   # cylinder ≈ h=1.2m, d=0.5m
    h_boiler = 5.0  # W/(m²·K)  well-insulated tank

    if T_b_old < net.T_supply:
        q_heat = net.heater_power
    elif T_b_old >= net.T_supply + 2.0:
        q_heat = 0.0
    else:
        q_heat = net.heater_power   # inside hysteresis band → stay on

    alpha_b = m_boiler * net.c / dt
    beta_b  = h_boiler * S_boiler
    mdot_out    = pipe_flow[0]
    mdot_return = net.mdot_recirc

    # T_cold is a known constant → its contribution goes to RHS (S_b) only.
    # T_ret is unknown → handled via implicit off-diagonal coupling below.
    a_P_b = alpha_b + beta_b + mdot_out * net.c
    S_b   = (alpha_b * T_b_old
             + beta_b * net.T_ambient
             + q_heat
             + mdot_cold_makeup * net.c * net.T_cold)

    return_gi = offsets[net.return_pipe_idx] + cells_per_pipe[net.return_pipe_idx] - 1
    add_entry(boiler_idx, return_gi, -mdot_return * net.c)
    a_P_b += mdot_return * net.c

    add_entry(boiler_idx, boiler_idx, a_P_b)
    b[boiler_idx] = S_b

    A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(N_total, N_total)).tocsr()
    return A, b, boiler_idx, offsets, pipe_flow


def simulate(net: DHWNetwork, t_end: float = 86400.0, dt: float = 10.0):
    """
    Run the implicit transient simulation.

    Parameters
    ----------
    net    : DHWNetwork  (pipe_flow_fn must be set)
    t_end  : float  total simulation time [s] (default 24 h)
    dt     : float  time step [s]

    Returns
    -------
    times     : 1-D array [s]
    T_history : 2-D array (n_snapshots, N_total)
    metadata  : dict with indexing info
    """
    cells_per_pipe = [p.N_cells for p in net.pipes]
    N_total = sum(cells_per_pipe) + 1

    T = np.full(N_total, net.T_ambient)
    T[-1] = net.T_supply   # boiler starts at setpoint

    n_steps = int(np.ceil(t_end / dt))
    save_every = max(1, n_steps // 2000)
    times, T_history = [], []

    for step in range(n_steps):
        t = step * dt
        A, b, boiler_idx, offsets, _ = build_global_system(net, T, t, dt)
        T = spsolve(A, b)

        if step % save_every == 0:
            times.append(t)
            T_history.append(T.copy())

    times = np.array(times)
    T_history = np.array(T_history)

    pipe_names = net.pipe_names if net.pipe_names else [f"pipe_{i}" for i in range(len(net.pipes))]

    metadata = {
        "offsets": offsets,
        "boiler_idx": boiler_idx,
        "cells_per_pipe": cells_per_pipe,
        "pipe_names": pipe_names,
        "return_pipe_idx": net.return_pipe_idx,
        "draw_node_names": [dn.name for dn in net.draw_nodes],
        "draw_node_global_idx": [
            offsets[dn.pipe_index] + dn.cell_index for dn in net.draw_nodes
        ],
        "hx_node_names": [hx.name for hx in net.hx_nodes],
        "hx_node_global_idx": [
            offsets[hx.pipe_index] + hx.cell_index for hx in net.hx_nodes
        ],
    }

    return times, T_history, metadata
