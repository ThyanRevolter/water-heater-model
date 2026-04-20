"""
Entry point for the DHW recirculation simulation.

Usage
-----
  python main.py                  # residential network (default)
  python main.py --industrial     # industrial facility network
"""

import argparse
from pathlib import Path

from dhw import (
    build_sample_network, build_industrial_network,
    simulate, plot_results, plot_boiler_power, plot_draw_flows, animate_temperatures,
    compute_energy_metrics, export_draw_csv, draw_network,
)

OUTPUT_DIR = Path("output")
PLOTS_DIR  = OUTPUT_DIR / "plots"
DATA_DIR   = OUTPUT_DIR / "data"


def main():
    parser = argparse.ArgumentParser(description="DHW recirculation simulation")
    parser.add_argument("--industrial", action="store_true",
                        help="Run the industrial facility network instead of residential")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.industrial:
        print("Building industrial DHW network...")
        net = build_industrial_network()
        stem = "dhw_industrial"
    else:
        print("Building residential DHW network...")
        net = build_sample_network()
        stem = "dhw_residential"

    print(f"  {len(net.pipes)} pipe segments, "
          f"{sum(p.N_cells for p in net.pipes)} cells + 1 boiler node")
    print(f"  {len(net.draw_nodes)} draw nodes, {len(net.hx_nodes)} HX nodes")
    print(f"  Recirculation flow : {net.mdot_recirc*1000:.1f} g/s")
    print(f"  Boiler setpoint    : {net.T_supply:.0f} °C")

    print("\nRunning 24-hour implicit simulation (dt=10 s)...")
    times, T_hist, meta = simulate(net, t_end=86400.0, dt=10.0)
    print(f"  Done. {len(times)} snapshots saved.")

    draw_network(net, save_path=PLOTS_DIR / f"{stem}_network.png")
    plot_results(times, T_hist, meta, net,
                 save_path=PLOTS_DIR / f"{stem}_results.png")
    plot_boiler_power(times, T_hist, meta, net,
                      save_path=PLOTS_DIR / f"{stem}_boiler_power.png")
    plot_draw_flows(times, meta, net,
                    save_path=PLOTS_DIR / f"{stem}_draw_flows.png")
    animate_temperatures(times, T_hist, meta, net,
                         save_path=PLOTS_DIR / f"{stem}_animation.gif")
    export_draw_csv(times, T_hist, meta, net,
                    save_path=DATA_DIR / f"{stem}_draw.csv")
    compute_energy_metrics(times, T_hist, meta, net)


if __name__ == "__main__":
    main()
