"""
Data classes representing the DHW network topology and parameters.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Optional


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
    """
    A point where water is drawn from the network.

    Governing addition to the cell energy balance:

        -mdot_draw * c * (T_i - T_cold)

    Hot water leaves at T_i; cold makeup enters at T_cold (mass balance).
    In implicit form this adds +mdot_draw*c to a_P and +mdot_draw*c*T_cold to S.
    """
    name: str
    pipe_index: int       # index into net.pipes
    cell_index: int       # cell along that pipe
    draw_profile: Callable  # f(t [s]) -> mass flow rate [kg/s]


@dataclass
class HeatExchangerNode:
    """
    A cell that transfers heat to a process load without drawing water.

    Governing addition to the cell energy balance:

        -UA_hx * (T_i - T_process)

    The process side is modelled as a fixed-temperature reservoir (T_process).
    In implicit form this adds +UA_hx to a_P and +UA_hx*T_process to S.

    UA_hx    : W/K  overall heat transfer conductance of the exchanger
    T_process: °C   process-side temperature (fixed boundary condition)
    """
    name: str
    pipe_index: int    # index into net.pipes
    cell_index: int    # cell along that pipe
    UA_hx: float       # W/K
    T_process: float   # °C


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
    hx_nodes: List[HeatExchangerNode] = field(default_factory=list)

    # Adjacency: pipes[i] feeds into pipes[j]
    adjacency: List[tuple] = field(default_factory=list)

    # parent_map[pipe_idx] = parent_pipe_idx whose last cell feeds pipe_idx's first cell.
    # Pipe 0 is always fed by the boiler (not in this map).
    parent_map: Dict[int, int] = field(default_factory=dict)

    # Index of the pipe whose last cell feeds the boiler return inlet.
    return_pipe_idx: int = 7

    # pipe_flow_fn(net, draw_at_pipe_cell) -> np.ndarray of shape (n_pipes,)
    # draw_at_pipe_cell: dict {(pipe_idx, cell_idx): mdot [kg/s]}
    pipe_flow_fn: Optional[Callable] = None

    # Human-readable names for each pipe (used in metadata / plots).
    pipe_names: List[str] = field(default_factory=list)
