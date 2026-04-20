"""
Factory functions for building DHW pipe networks.

Spatial discretisation
----------------------
Both networks use a fixed target cell size `dz` so that spatial resolution
is uniform across pipes of different lengths:

    N_cells = max(1, round(pipe.length / dz))

HX node positions are specified as physical distances from the pipe inlet
and converted to cell indices automatically.

Networks
--------
build_sample_network()      — residential recirculation loop (8 pipes, 5 draw nodes)
build_industrial_network()  — industrial facility loop (6 pipes, 2 HX nodes, 3 draw nodes)
"""

import numpy as np
import pandas as pd
from typing import Callable
from .models import PipeParams, DrawNode, HeatExchangerNode, DHWNetwork


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _n_cells(length: float, dz: float) -> int:
    """Number of cells that gives approximately uniform cell size dz."""
    return max(1, round(length / dz))


def _cell_at(position_m: float, pipe: PipeParams) -> int:
    """Cell index corresponding to a physical distance from the pipe inlet."""
    dz = pipe.length / pipe.N_cells
    return min(pipe.N_cells - 1, int(position_m / dz))


def default_pipe(length=5.0, dz=0.5, d_inner=0.02, insul_t=0.025) -> PipeParams:
    """Residential pipe (copper, foam insulation) with cell size ≈ dz metres."""
    return PipeParams(
        length=length,
        inner_diameter=d_inner,
        wall_thickness=0.001,          # 1 mm copper
        insulation_k=0.035,            # W/(m·K) foam
        insulation_t=insul_t,
        h_ext=10.0,                    # W/(m²·K) free convection
        pipe_k=385.0,                  # copper
        N_cells=_n_cells(length, dz),
    )


def _make_draw_profile(peak_kg_s, on_hours):
    """Return f(t_seconds) -> mdot [kg/s] as a step function."""
    def profile(t):
        hour = (t / 3600.0) % 24.0
        for h_start, h_end in on_hours:
            if h_start <= hour < h_end:
                return peak_kg_s
        return 0.0
    return profile


def _read_draw_profile(path: str) -> Callable:
    """Read a draw profile from a CSV file."""
    profile = pd.read_csv(path)
    def profile(t):
        return profile.loc[profile['time'] == t, 'mdot'].values[0]
    return profile


# ──────────────────────────────────────────────────────────────
# Residential network
# ──────────────────────────────────────────────────────────────

def _residential_pipe_flow(net: DHWNetwork, draw_at_pipe_cell: dict) -> np.ndarray:
    """
    Hydraulic model for the residential 8-pipe network.

    Flow assignment (upstream → downstream):
      pipe 7 (return)   : mdot_recirc
      pipe 6 (kitchen)  : mdot_recirc + kitchen draw
      pipe 4,5 (bath2)  : their own draws only
      pipe 3 (riser F2) : sum of pipes 4, 5, 6
      pipe 1,2 (bath1)  : their own draws only
      pipe 0 (supply)   : sum of pipes 1, 2, 3
    """
    cpc = [p.N_cells for p in net.pipes]
    q = np.zeros(len(net.pipes))
    q[7] = net.mdot_recirc
    q[6] = net.mdot_recirc + draw_at_pipe_cell.get((6, cpc[6]-1), 0.0)
    q[4] = draw_at_pipe_cell.get((4, cpc[4]-1), 0.0)
    q[5] = draw_at_pipe_cell.get((5, cpc[5]-1), 0.0)
    q[3] = q[4] + q[5] + q[6]
    q[1] = draw_at_pipe_cell.get((1, cpc[1]-1), 0.0)
    q[2] = draw_at_pipe_cell.get((2, cpc[2]-1), 0.0)
    q[0] = q[1] + q[2] + q[3]
    return q


def build_sample_network(dz: float = 0.5) -> DHWNetwork:
    """
    Residential DHW recirculation loop (two-storey house).

    All pipes use a fixed cell size of approximately dz metres, so longer
    pipes have more cells and shorter pipes have fewer — giving uniform
    spatial resolution across the network.

    Topology
    --------
    Boiler → supply_main → junction_F1 ─┬─> bath1_shower  [draw]
                                         ├─> bath1_faucet  [draw]
                                         └─> riser_F2 ────> junction_F2
                                                  ├─> bath2_shower  [draw]
                                                  ├─> bath2_faucet  [draw]
                                                  └─> kitchen_sink  [draw]
                                                        └─> recirc_return → Boiler

    Pipe index map
    --------------
    0  supply_main      8 m   Boiler → Junction_F1
    1  bath1_shower     3 m   Junction_F1 → Bath1 shower  (draw at outlet)
    2  bath1_faucet     2 m   Junction_F1 → Bath1 faucet  (draw at outlet)
    3  riser_F2         4 m   Junction_F1 → Junction_F2
    4  bath2_shower     3 m   Junction_F2 → Bath2 shower  (draw at outlet)
    5  bath2_faucet     2 m   Junction_F2 → Bath2 faucet  (draw at outlet)
    6  kitchen_sink     5 m   Junction_F2 → Kitchen sink  (draw at outlet)
    7  recirc_return   12 m   Kitchen end → Boiler
    """
    net = DHWNetwork(
        return_pipe_idx=7,
        pipe_flow_fn=_residential_pipe_flow,
    )

    pipe_lengths = [8.0, 3.0, 2.0, 4.0, 3.0, 2.0, 5.0, 12.0]
    net.pipes = [
        default_pipe(length=l, dz=dz)
        for l in pipe_lengths
    ]
    # Narrower, less insulated return pipe
    net.pipes[7] = default_pipe(length=12.0, dz=dz, d_inner=0.012, insul_t=0.02)

    net.pipe_names = [
        "supply_main", "bath1_shower", "bath1_faucet",
        "riser_F2", "bath2_shower", "bath2_faucet",
        "kitchen", "recirc_return",
    ]
    net.parent_map = {1: 0, 2: 0, 3: 0, 4: 3, 5: 3, 6: 3, 7: 6}
    net.adjacency  = [(0,1),(0,2),(0,3),(3,4),(3,5),(3,6),(6,7)]

    # Draw nodes at the outlet (last cell) of each branch pipe
    cpc = [p.N_cells for p in net.pipes]
    net.draw_nodes = [
        DrawNode("Bathroom1_Shower", 1, cpc[1]-1,
                 _make_draw_profile(0.15, [(7.0, 7.25), (22.0, 22.15)])),
        DrawNode("Bathroom1_Faucet", 2, cpc[2]-1,
                 _make_draw_profile(0.05, [(7.25, 7.33), (12.0, 12.08)])),
        DrawNode("Bathroom2_Shower", 4, cpc[4]-1,
                 _make_draw_profile(0.15, [(7.5, 7.75)])),
        DrawNode("Bathroom2_Faucet", 5, cpc[5]-1,
                 _make_draw_profile(0.03, [(8.0, 8.08), (18.0, 18.08)])),
        DrawNode("Kitchen_Sink",     6, cpc[6]-1,
                 _make_draw_profile(0.10, [(7.0, 7.08), (12.0, 12.17),
                                           (18.0, 18.33)])),
    ]

    _print_discretisation(net, dz, "Residential")
    return net


# ──────────────────────────────────────────────────────────────
# Industrial network
# ──────────────────────────────────────────────────────────────

def _industrial_pipe(length, d_inner, dz=1.0, insul_t=0.04) -> PipeParams:
    """Industrial pipe (stainless steel, mineral wool) with cell size ≈ dz metres."""
    return PipeParams(
        length=length,
        inner_diameter=d_inner,
        wall_thickness=0.003,          # 3 mm stainless
        insulation_k=0.04,             # W/(m·K) mineral wool
        insulation_t=insul_t,
        h_ext=15.0,                    # W/(m²·K) light forced convection (plant floor)
        pipe_k=16.0,                   # stainless steel
        N_cells=_n_cells(length, dz),
    )


def _industrial_pipe_flow(net: DHWNetwork, draw_at_pipe_cell: dict) -> np.ndarray:
    """
    Hydraulic model for the industrial 6-pipe network.

    The recirculation pump drives all flow through the HX loop (pipe 1).
    Draw branches carry only their demand; no recirc sub-loop through them.

    Flow assignment
    ---------------
      pipe 5 (return header) : mdot_recirc
      pipe 1 (HX loop)       : mdot_recirc  (full recirc through HX)
      pipe 2 (showers)       : shower draw
      pipe 3 (break room)    : break room draw
      pipe 4 (process draw)  : process draw
      pipe 0 (supply header) : sum of all branch flows
    """
    cpc = [p.N_cells for p in net.pipes]
    q = np.zeros(len(net.pipes))
    q[5] = net.mdot_recirc
    q[1] = net.mdot_recirc
    q[2] = draw_at_pipe_cell.get((2, cpc[2]-1), 0.0)
    q[3] = draw_at_pipe_cell.get((3, cpc[3]-1), 0.0)
    q[4] = draw_at_pipe_cell.get((4, cpc[4]-1), 0.0)
    q[0] = q[1] + q[2] + q[3] + q[4]
    return q


def build_industrial_network(dz: float = 1.0) -> DHWNetwork:
    """
    Industrial DHW recirculation loop for a manufacturing facility.

    All pipes use a fixed cell size of approximately dz metres.  HX node
    positions are given as physical distances from the pipe inlet and
    automatically converted to cell indices.

    Topology
    --------
    Boiler → supply_header ─┬─> hx_loop → [HX1 @ 10 m] → [HX2 @ 22.5 m] → return_header → Boiler
             (pipe 0)        ├─> locker_showers  [draw at outlet]
                             ├─> breakroom       [draw at outlet]
                             └─> process_wash    [draw at outlet]

    Pipe index map
    --------------
    0  supply_header   20 m  DN50  Boiler → branch junction
    1  hx_loop         30 m  DN40  junction → HX1 → HX2 → return junction
    2  locker_showers  15 m  DN32  junction → shower block   (draw at outlet)
    3  breakroom        8 m  DN20  junction → break room      (draw at outlet)
    4  process_wash    10 m  DN32  junction → process station (draw at outlet)
    5  return_header   18 m  DN40  return junction → Boiler

    HX nodes (pipe 1, hx_loop)
    ---------------------------
    HX1  PartsWasher   10.0 m from inlet   UA=600 W/K   T_proc=45 °C
    HX2  AirHandler    22.5 m from inlet   UA=350 W/K   T_proc=35 °C

    Boiler / system parameters
    --------------------------
    500 L tank, 30 kW element, T_supply=70 °C, mdot_recirc=0.05 kg/s
    """
    net = DHWNetwork(
        T_supply=70.0,
        heater_power=30_000.0,
        boiler_volume=0.500,
        mdot_recirc=0.05,
        T_ambient=25.0,
        return_pipe_idx=5,
        pipe_flow_fn=_industrial_pipe_flow,
    )

    net.pipes = [
        _industrial_pipe(length=20.0, d_inner=0.050, dz=dz),   # 0 supply header
        _industrial_pipe(length=30.0, d_inner=0.040, dz=dz),   # 1 HX loop
        _industrial_pipe(length=15.0, d_inner=0.032, dz=dz),   # 2 locker showers
        _industrial_pipe(length=8.0,  d_inner=0.020, dz=dz),   # 3 break room
        _industrial_pipe(length=10.0, d_inner=0.032, dz=dz),   # 4 process wash
        _industrial_pipe(length=18.0, d_inner=0.040, dz=dz),   # 5 return header
    ]
    net.pipe_names = [
        "supply_header", "hx_loop",
        "locker_showers", "breakroom", "process_wash",
        "return_header",
    ]
    net.parent_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
    net.adjacency  = [(0,1),(0,2),(0,3),(0,4),(1,5)]

    # HX nodes: positions given in metres from the hx_loop inlet
    hx_loop = net.pipes[1]
    net.hx_nodes = [
        HeatExchangerNode(
            name="PartsWasher_HX",
            pipe_index=1,
            cell_index=_cell_at(10.0, hx_loop),   # 10 m from inlet
            UA_hx=600.0,
            T_process=45.0,
        ),
        HeatExchangerNode(
            name="AirHandler_HX",
            pipe_index=1,
            cell_index=_cell_at(22.5, hx_loop),   # 22.5 m from inlet
            UA_hx=350.0,
            T_process=35.0,
        ),
    ]

    # Draw nodes at the outlet (last cell) of each branch pipe
    cpc = [p.N_cells for p in net.pipes]
    net.draw_nodes = [
        DrawNode("Locker_Showers", 2, cpc[2]-1,
                 _make_draw_profile(0.30, [(6.0, 6.5), (14.0, 14.5), (22.0, 22.5)])),
        DrawNode("Breakroom_Sink", 3, cpc[3]-1,
                 _make_draw_profile(0.04, [(10.0, 10.25), (12.0, 12.5), (15.0, 15.25)])),
        DrawNode("Process_Wash",   4, cpc[4]-1,
                 _make_draw_profile(0.12, [(7.0, 14.0), (15.0, 22.0)])),
    ]

    _print_discretisation(net, dz, "Industrial")
    return net


# ──────────────────────────────────────────────────────────────
# Diagnostic helper
# ──────────────────────────────────────────────────────────────

def _print_discretisation(net: DHWNetwork, dz: float, label: str):
    print(f"  [{label}] target dz={dz} m — pipe discretisation:")
    for name, pipe in zip(net.pipe_names, net.pipes):
        actual_dz = pipe.length / pipe.N_cells
        print(f"    {name:<20s}  L={pipe.length:5.1f} m  "
              f"N={pipe.N_cells:3d} cells  dz={actual_dz:.3f} m")
