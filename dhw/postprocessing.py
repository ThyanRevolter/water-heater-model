"""
Post-processing: plotting, energy metrics, CSV export, and network graph for DHW simulations.
"""

import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from .models import DHWNetwork
from .solver import compute_pipe_UA


def plot_results(times, T_history, metadata, net: DHWNetwork, save_path="dhw_results.png"):
    """Generate and save the three-panel results plot."""
    hours = times / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (a) Boiler temperature
    ax = axes[0]
    ax.plot(hours, T_history[:, metadata["boiler_idx"]], "r-", linewidth=1.5)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("(a) Boiler Tank Temperature")
    ax.axhline(net.T_supply, color="gray", linestyle="--", linewidth=0.8,
               label=f"Setpoint {net.T_supply:.0f} °C")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Draw-node delivery temperatures
    ax = axes[1]
    for name, gi in zip(metadata["draw_node_names"], metadata["draw_node_global_idx"]):
        ax.plot(hours, T_history[:, gi], linewidth=1.2, label=name)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("(b) Water Temperature at Draw-off Points")
    ax.axhline(40, color="gray", linestyle="--", linewidth=0.8, label="Min comfort 40 °C")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) Recirculation return vs boiler
    ax = axes[2]
    offsets = metadata["offsets"]
    rpi = metadata["return_pipe_idx"]
    ret_idx = offsets[rpi] + metadata["cells_per_pipe"][rpi] - 1
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


def plot_boiler_power(times, T_history, metadata, net: DHWNetwork,
                      save_path="dhw_boiler_power.png"):
    """
    Plot the boiler heating element power [kW] over the simulation period.

    The on/off state is reconstructed from saved boiler temperatures using
    the same hysteresis logic as the solver (ON below T_supply, OFF above
    T_supply + 2 °C).
    """
    hours   = times / 3600.0
    T_boiler = T_history[:, metadata["boiler_idx"]]

    # Reconstruct thermostat state with hysteresis
    power_kW = np.zeros(len(times))
    state = T_boiler[0] < net.T_supply
    for i, T_b in enumerate(T_boiler):
        if T_b < net.T_supply:
            state = True
        elif T_b >= net.T_supply + 2.0:
            state = False
        power_kW[i] = net.heater_power / 1000.0 if state else 0.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hours, power_kW, step="post", alpha=0.35, color="#e63946")
    ax.step(hours, power_kW, where="post", color="#e63946", linewidth=1.5,
            label=f"Heater ({net.heater_power/1000:.0f} kW rated)")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Boiler Heating Element Power Draw")
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(bottom=0)
    ax.axhline(net.heater_power / 1000.0, color="gray", linestyle="--",
               linewidth=0.8, label="Rated power")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Boiler power plot saved to {save_path}")


def plot_draw_flows(times, metadata, net: DHWNetwork, save_path="dhw_draw_flows.png"):
    """Plot mass flow rate at every draw node over the simulation period."""
    hours = times / 3600.0
    draw_nodes = net.draw_nodes

    fig, ax = plt.subplots(figsize=(12, 5))

    for dn in draw_nodes:
        mdot = np.array([dn.draw_profile(t) * 1000 for t in times])  # g/s
        ax.step(hours, mdot, where="post", linewidth=1.5, label=dn.name)

    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Flow rate [g/s]")
    ax.set_title("Draw Node Mass Flow Rates")
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Draw flow plot saved to {save_path}")


def animate_temperatures(times, T_history, metadata, net: DHWNetwork,
                         save_path="dhw_animation.gif", fps=24, n_frames=480):
    """
    Animate how temperature evolves at every node over the simulation period.

    Layout
    ------
    Top panel   — heatmap: rows = pipes, columns = cells, colour = temperature.
                  Triangles mark draw nodes; squares mark HX nodes.
    Bottom panel — boiler temperature trace with a moving time cursor.

    Output
    ------
    Tries MP4 (ffmpeg) first; falls back to GIF (pillow).
    """
    offsets        = metadata["offsets"]
    cells_per_pipe = metadata["cells_per_pipe"]
    boiler_idx     = metadata["boiler_idx"]
    pipe_names     = metadata["pipe_names"]
    n_pipes        = len(net.pipes)

    # ── Downsample to n_frames evenly spaced snapshots ──
    idx      = np.linspace(0, len(times) - 1, min(n_frames, len(times)), dtype=int)
    T_frames = T_history[idx]
    t_frames = times[idx] / 3600.0   # hours
    hours    = times / 3600.0

    # ── Extract pipe temperatures: (n_frames, n_pipes, max_cells) ──
    N_max = max(cells_per_pipe)
    pipe_T = np.full((len(idx), n_pipes, N_max), np.nan)
    for p in range(n_pipes):
        nc = cells_per_pipe[p]
        pipe_T[:, p, :nc] = T_frames[:, offsets[p]: offsets[p] + nc]

    boiler_T = T_frames[:, boiler_idx]

    T_min = min(net.T_cold, net.T_ambient) - 2
    T_max = net.T_supply + 5

    # ── Figure ──
    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_map    = fig.add_subplot(gs[0])
    ax_boiler = fig.add_subplot(gs[1])

    # Heatmap (top)
    im = ax_map.imshow(
        pipe_T[0], aspect="auto", origin="upper",
        vmin=T_min, vmax=T_max, cmap="plasma",
    )
    cbar = fig.colorbar(im, ax=ax_map, label="Temperature [°C]", shrink=0.9, pad=0.02)
    ax_map.set_yticks(range(n_pipes))
    ax_map.set_yticklabels(pipe_names, fontsize=9)
    ax_map.set_xlabel("Cell index →  (left = inlet, right = outlet)", fontsize=9)
    title = ax_map.set_title(f"Network Temperature  —  t = 0.00 h", fontsize=11)

    # Node markers on the heatmap
    marker_handles = []
    for dn in net.draw_nodes:
        h = ax_map.plot(dn.cell_index, dn.pipe_index, "v",
                        color="cyan", markersize=9, markeredgecolor="k", linewidth=0.5)[0]
    for hx in net.hx_nodes:
        h = ax_map.plot(hx.cell_index, hx.pipe_index, "s",
                        color="lime", markersize=9, markeredgecolor="k", linewidth=0.5)[0]

    if net.draw_nodes:
        marker_handles.append(mpatches.Patch(color="cyan",  label="Draw node"))
    if net.hx_nodes:
        marker_handles.append(mpatches.Patch(color="lime",  label="HX node"))
    if marker_handles:
        ax_map.legend(handles=marker_handles, loc="upper right", fontsize=8)

    # Boiler trace (bottom)
    ax_boiler.plot(hours, T_history[:, boiler_idx],
                   color="#e63946", linewidth=1.2, alpha=0.35)
    ax_boiler.axhline(net.T_supply, color="gray", linestyle="--",
                      linewidth=0.8, label=f"Setpoint {net.T_supply:.0f} °C")
    cursor_dot,  = ax_boiler.plot([], [], "o", color="#e63946", markersize=7, zorder=5)
    cursor_line  = ax_boiler.axvline(0, color="#e63946", linewidth=1.0, alpha=0.6)
    ax_boiler.set_xlim(0, hours[-1])
    ax_boiler.set_ylim(T_min, T_max)
    ax_boiler.set_xlabel("Time [hours]", fontsize=9)
    ax_boiler.set_ylabel("Boiler T [°C]", fontsize=9)
    ax_boiler.legend(fontsize=8, loc="upper right")
    ax_boiler.grid(True, alpha=0.3)

    # ── Update function ──
    def update(frame):
        im.set_data(pipe_T[frame])
        title.set_text(f"Network Temperature  —  t = {t_frames[frame]:.2f} h")
        cursor_dot.set_data([t_frames[frame]], [boiler_T[frame]])
        cursor_line.set_xdata([t_frames[frame]])
        return im, title, cursor_dot, cursor_line

    anim = animation.FuncAnimation(
        fig, update, frames=len(idx), interval=1000 / fps, blit=True,
    )

    # ── Save: try MP4, fall back to GIF ──
    try:
        mp4_path = str(save_path).replace(".gif", ".mp4")
        anim.save(mp4_path, writer="ffmpeg", fps=fps, dpi=120,
                  extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        save_path = mp4_path
    except Exception:
        gif_path = str(save_path).replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        save_path = gif_path

    plt.close()
    print(f"Animation saved to {save_path}")


def export_draw_csv(times, T_history, metadata, net: DHWNetwork, save_path="dhw_draw.csv"):
    """
    Write a CSV with one row per saved timestep containing:
      time_s, time_h,
      for each draw node: <name>_mdot_kg_s, <name>_temp_C
    """
    draw_names = metadata["draw_node_names"]
    draw_gidx  = metadata["draw_node_global_idx"]
    draw_nodes = net.draw_nodes

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["time_s", "time_h"]
        for name in draw_names:
            header += [f"{name}_mdot_kg_s", f"{name}_temp_C"]
        writer.writerow(header)

        for i, t in enumerate(times):
            row = [f"{t:.1f}", f"{t/3600:.4f}"]
            for dn, gi in zip(draw_nodes, draw_gidx):
                mdot = dn.draw_profile(t)
                temp = T_history[i, gi]
                row += [f"{mdot:.4f}", f"{temp:.4f}"]
            writer.writerow(row)

    print(f"Draw CSV saved to {save_path}")


def draw_network(net: DHWNetwork, save_path="network.png"):
    """
    Draw the pipe network topology using NetworkX.

    Nodes
    -----
    boiler          — red square
    pipe segments   — grey circles
    draw nodes      — blue circles
    HX nodes        — orange circles

    Edges
    -----
    Directed arrows follow flow direction.
    Edge labels show pipe length [m] and inner diameter [mm].
    """
    G = nx.DiGraph()
    pipe_names = net.pipe_names if net.pipe_names else [f"pipe_{i}" for i in range(len(net.pipes))]

    # ── Nodes ──
    G.add_node("boiler", kind="boiler")
    for name in pipe_names:
        G.add_node(name, kind="pipe")
    for dn in net.draw_nodes:
        G.add_node(dn.name, kind="draw")
    for hx in net.hx_nodes:
        G.add_node(hx.name, kind="hx")

    # ── Edges ──
    pipe_edge_labels = {}

    p0 = net.pipes[0]
    G.add_edge("boiler", pipe_names[0])
    pipe_edge_labels[("boiler", pipe_names[0])] = f"{p0.length:.0f} m / DN{p0.inner_diameter*1000:.0f}"

    for child, parent in net.parent_map.items():
        p = net.pipes[child]
        G.add_edge(pipe_names[parent], pipe_names[child])
        pipe_edge_labels[(pipe_names[parent], pipe_names[child])] = f"{p.length:.0f} m / DN{p.inner_diameter*1000:.0f}"

    G.add_edge(pipe_names[net.return_pipe_idx], "boiler")

    for dn in net.draw_nodes:
        G.add_edge(pipe_names[dn.pipe_index], dn.name)
    for hx in net.hx_nodes:
        G.add_edge(pipe_names[hx.pipe_index], hx.name)

    # ── Hierarchical layout ──
    pos = _hierarchical_pos(G, "boiler")

    # ── Draw ──
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(
        f"DHW Network Topology — {len(net.pipes)} pipes, "
        f"{len(net.draw_nodes)} draw nodes, {len(net.hx_nodes)} HX nodes",
        fontsize=13, pad=14,
    )

    node_colors = {"boiler": "#e63946", "pipe": "#6c757d", "draw": "#4361ee", "hx": "#f4a261"}
    node_sizes  = {"boiler": 2000, "pipe": 1800, "draw": 1600, "hx": 1600}

    for kind in ("boiler", "pipe", "draw", "hx"):
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_color=node_colors[kind],
            node_shape="s" if kind == "boiler" else "o",
            node_size=node_sizes[kind],
        )

    # Display labels: replace underscores with newlines, split CamelCase runs > 8 chars
    def _wrap(name):
        # underscores → newlines
        parts = name.split("_")
        # if any single part is still long, break at 8 chars
        wrapped = []
        for p in parts:
            if len(p) > 8:
                wrapped += [p[i:i+8] for i in range(0, len(p), 8)]
            else:
                wrapped.append(p)
        return "\n".join(wrapped)

    labels = {n: _wrap(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color="white", font_weight="bold")

    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True,
        arrowstyle="-|>", arrowsize=20,
        edge_color="#343a40", width=1.8,
        connectionstyle="arc3,rad=0.06",
        min_source_margin=25, min_target_margin=25,
    )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=pipe_edge_labels, ax=ax,
        font_size=7, label_pos=0.45,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )

    legend_handles = [
        mpatches.Patch(color=node_colors["boiler"], label="Boiler"),
        mpatches.Patch(color=node_colors["pipe"],   label="Pipe segment"),
        mpatches.Patch(color=node_colors["draw"],   label="Draw node"),
        mpatches.Patch(color=node_colors["hx"],     label="Heat exchanger"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10, framealpha=0.9)
    ax.axis("off")
    ax.margins(0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Network diagram saved to {save_path}")


def _hierarchical_pos(G: nx.DiGraph, root: str, x_gap=2.5, y_gap=1.8):
    """
    Compute a top-down hierarchical (tree) layout via BFS from root.
    Nodes not reachable from root (e.g. return-pipe → boiler back-edge)
    are placed at the bottom.
    """
    # BFS to assign layers
    layers = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for successor in G.successors(node):
            if successor not in layers:
                layers[successor] = layers[node] + 1
                queue.append(successor)

    # Group nodes by layer
    from collections import defaultdict
    layer_nodes = defaultdict(list)
    for node, layer in layers.items():
        layer_nodes[layer].append(node)

    pos = {}
    for layer, nodes in sorted(layer_nodes.items()):
        n = len(nodes)
        xs = [(i - (n - 1) / 2) * x_gap for i in range(n)]
        for node, x in zip(nodes, xs):
            pos[node] = (x, -layer * y_gap)

    # Any node not yet placed (disconnected in BFS direction)
    max_layer = max(layer_nodes.keys()) if layer_nodes else 0
    unplaced = [n for n in G.nodes() if n not in pos]
    for i, node in enumerate(unplaced):
        pos[node] = ((i - len(unplaced) / 2) * x_gap, -(max_layer + 1) * y_gap)

    return pos


def compute_energy_metrics(times, T_history, metadata, net: DHWNetwork):
    """Compute and print energy performance metrics."""
    dt_arr = np.diff(times)
    T_boiler = T_history[:, metadata["boiler_idx"]]

    heater_on = T_boiler[:-1] < net.T_supply
    energy_heater_kWh = np.sum(heater_on * net.heater_power * dt_arr) / 3.6e6

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

    avg_loss_W = np.mean(total_loss_W)

    print("\n" + "=" * 55)
    print("  DHW RECIRCULATION SYSTEM — ENERGY SUMMARY")
    print("=" * 55)
    print(f"  Simulation period     : {times[-1]/3600:.1f} hours")
    print(f"  Heater energy input   : {energy_heater_kWh:.2f} kWh")
    print(f"  Avg pipe heat loss    : {avg_loss_W:.1f} W")
    print(f"  Recirc pump flow      : {net.mdot_recirc*1000:.1f} g/s")
    print("=" * 55 + "\n")
