"""
dhw — Domestic Hot Water recirculation thermal-hydraulic model.

Modules
-------
models          : PipeParams, DrawNode, DHWNetwork dataclasses
network         : default_pipe(), build_sample_network()
solver          : compute_pipe_UA(), build_global_system(), simulate()
postprocessing  : plot_results(), compute_energy_metrics()
"""

from .models import PipeParams, DrawNode, HeatExchangerNode, DHWNetwork
from .network import default_pipe, build_sample_network, build_industrial_network
from .solver import compute_pipe_UA, build_global_system, simulate
from .postprocessing import plot_results, plot_boiler_power, plot_draw_flows, animate_temperatures, compute_energy_metrics, export_draw_csv, draw_network
from .pypes_network import DHWDrawNetwork

__all__ = [
    "PipeParams",
    "DrawNode",
    "HeatExchangerNode",
    "DHWNetwork",
    "default_pipe",
    "build_sample_network",
    "build_industrial_network",
    "compute_pipe_UA",
    "build_global_system",
    "simulate",
    "plot_results",
    "compute_energy_metrics",
    "plot_boiler_power",
    "plot_draw_flows",
    "animate_temperatures",
    "export_draw_csv",
    "draw_network",
    "DHWDrawNetwork",
]
