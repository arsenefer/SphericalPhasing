"""Animate the antenna-layout plot while sweeping phi from left to right.

The animation keeps the event geometry fixed and recomputes the ADF-predicted
amplitudes for each frame as phi changes. The shower direction arrow rotates
with phi, and the antenna marker colors/sizes follow the predicted amplitudes.

Usage
-----
python adf_analysis/animate_adf_plot.py --event-id 40 --output plot_slides/adf_phi_animation.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append("/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import beamforming_module.beamforming_module_para as bm
import beamforming_module.read_sims_module as rm
from beamforming_module.adf_weights import ADF_parameters, shower_direction_vector


R2D = 180.0 / np.pi


class Config:
    path_to_library: str = (
        "/volatile/home/af274537/Documents/DATA/HERON/"
        "TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    )
    f_min: float = 30.0
    f_max: float = 250.0
    noise_std: float = 13.0
    jitter_std: float = 0.0


def load_event(event_id: int, config: Config):
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    (
        azim,
        zen,
        en_nu,
        en_tau,
        tau_pos,
        xmax_pos,
        event_efield,
        antenna_pos,
        xant,
        yant,
        zant,
        signals,
        phi,
        theta,
        shower_dir,
    ) = rm.read_trace(
        event_id,
        tau_events,
        antennas,
        efields,
        sttimes,
        config.f_min,
        config.f_max,
        config.noise_std,
        config.jitter_std,
    )

    return {
        "event_id": event_id,
        "tau_pos": tau_pos,
        "xmax_pos": xmax_pos,
        "antenna_pos": antenna_pos,
        "xant": xant,
        "yant": yant,
        "zant": zant,
        "signals": signals,
        "phi": phi,
        "theta": theta,
        "shower_dir": shower_dir,
    }


def predicted_amplitudes(theta: float, phi: float, xmax_pos: np.ndarray, antennas_xyz: np.ndarray, delta_omega: float):
    _, _, _, _, adf = ADF_parameters(
        theta,
        phi,
        delta_omega=delta_omega,
        Xants=antennas_xyz,
        Xsource=xmax_pos,
        norm_weights=False,
    )
    return np.asarray(adf)


def build_animation(event, output_path: Path, phi_span_deg: float, frames: int, delta_omega: float, pause_frames: int = 12):
    signals = event["signals"]
    xant_true = event["xant"]
    yant_true = event["yant"]
    zant_true = event["zant"]
    xmax_pos_true = event["xmax_pos"]
    tau_pos_true = event["tau_pos"]

    xant_vis = xant_true
    yant_vis = yant_true * 10
    zant_vis = zant_true
    xmax_pos_vis = np.array([3e3, -40000, xmax_pos_true[2]])  # visualization position
    tau_pos_vis = np.array([3e3, -55000, tau_pos_true[2]])  # visualization position
    theta = event["theta"]
    phi_center_vis = -np.pi / 2
    phi_center_true = event["phi"]
    shower_dir = event["shower_dir"]

    antennas_xyz_vis = np.stack((xant_vis, yant_vis, zant_vis), axis=-1)
    antennas_xyz_true = np.stack((xant_true, yant_true, zant_true), axis=-1)
    shifted_times_true = bm.spherical_phasing(
        xmax_pos_true[0], xmax_pos_true[1], xmax_pos_true[2],
        xant_true, yant_true, zant_true, signals[0, :, 0]
    )

    phi_offsets = np.linspace(-phi_span_deg / R2D, phi_span_deg / R2D, frames)
    phi_values_sweep_vis = phi_center_vis + phi_offsets
    phi_values_sweep_true = phi_center_true + phi_offsets
    center_idx = len(phi_offsets) // 2
    # insert a short pause at the center of the sweep (when phi == center)
    if pause_frames > 0:
        phi_values = np.concatenate((
            phi_values_sweep_vis[: center_idx + 1],
            np.repeat(phi_values_sweep_vis[center_idx], pause_frames),
            phi_values_sweep_vis[center_idx + 1 :],
        ))
    else:
        phi_values = phi_values_sweep_vis
    phi_values_deg = (phi_values - phi_center_vis) * R2D

    # Precompute ADF amplitudes across the sweep and the matching weighted sums
    adf_sweep = []
    beam_time_axis = None
    beam_trace_sweep = []
    for phi_vis_val, phi_true_val in zip(phi_values_sweep_vis, phi_values_sweep_true):
        _, _, _, _, weights = ADF_parameters(
            theta,
            phi_vis_val,
            delta_omega=delta_omega,
            Xants=antennas_xyz_vis,
            Xsource=xmax_pos_vis,
            norm_weights=False,
        )
        adf_sweep.append(weights)

        _, _, _, _, beam_weights = ADF_parameters(
            theta,
            phi_true_val,
            delta_omega=delta_omega,
            Xants=antennas_xyz_true,
            Xsource=xmax_pos_true,
            norm_weights=True,
        )
        beam_output = np.array(bm.signals_summing_adf(signals, shifted_times_true, weights=beam_weights[None, :]))[0]
        if beam_time_axis is None:
            beam_time_axis = beam_output[0]
        beam_trace_sweep.append(beam_output)

    adf_sweep = np.vstack(adf_sweep)
    beam_trace_sweep = np.asarray(beam_trace_sweep)
    if pause_frames > 0:
        adf_matrix = np.vstack((
            adf_sweep[: center_idx + 1],
            np.repeat(adf_sweep[center_idx][None, :], pause_frames, axis=0),
            adf_sweep[center_idx + 1 :],
        ))
        beam_trace_matrix = np.concatenate((
            beam_trace_sweep[: center_idx + 1],
            np.repeat(beam_trace_sweep[center_idx][None, :, :], pause_frames, axis=0),
            beam_trace_sweep[center_idx + 1 :],
        ))
    else:
        adf_matrix = adf_sweep
        beam_trace_matrix = beam_trace_sweep
    adf_vmin = float(np.nanmin(adf_matrix))
    adf_vmax = float(np.nanmax(adf_matrix))
    # initial colors = center frame
    initial_adf = adf_matrix[center_idx]

    # Visual style: larger labels and title
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, (ax, ax_sig) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle("ADF sweeping", fontsize=16)

    # compute static edge colors using ADF at the true phi (phi_center)
    adf_true = adf_matrix[center_idx]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=adf_vmin, vmax=adf_vmax)
    edge_colors = cmap(norm(adf_true))

    # fixed marker size; color shows ADF-predicted amplitude, edgecolors reflect true-phi ADF
    scatter_xy = ax.scatter(
        xant_vis / 1e3,
        yant_vis / 1e3,
        c=initial_adf,
        cmap="viridis",
        s=80,
        vmin=adf_vmin,
        vmax=adf_vmax,
        edgecolors=edge_colors,
        linewidths=2,
    )
    ax.scatter(xmax_pos_vis[0] / 1e3, xmax_pos_vis[1] / 1e3, c="red", marker="*", s=200, label="Xmax")

    initial_end = xmax_pos_vis + shower_dir * np.linalg.norm(tau_pos_vis)
    line_xy, = ax.plot(
        [xmax_pos_vis[0] / 1e3, initial_end[0] / 1e3],
        [xmax_pos_vis[1] / 1e3, initial_end[1] / 1e3],
        c="blue",
        linestyle="--",
        label="Shower direction",
    )

    # move dynamic info text to bottom center of the figure
    info_text = ax.text(
        0.5,
        0.03,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # big center overlay for success message (hidden until pause)
    success_text = fig.text(
        0.5,
        0.72,
        "",
        ha="center",
        va="center",
        fontsize=28,
        color="green",
        weight="bold",
        alpha=0.0,
    )

    ax.set_xlabel("X Position (km)")
    ax.set_ylabel("Y Position (km)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=2, framealpha=0.9)

    ax_sig.set_title("ADF weighted sum")
    ax_sig.set_xlabel("Time (ns)")
    ax_sig.set_ylabel("Summed amp. (µV/m)")
    ax_sig.grid(alpha=0.2)

    # reserve whitespace on the right for colorbar and bottom-plot legend
    right_reserved = 0.82
    fig.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08, right=1.)

    # Ensure the bottom axis doesn't extend into the reserved right whitespace.
    # Shrink ax_sig width if necessary while keeping its left edge.
    pos_sig = ax_sig.get_position()
    current_left = pos_sig.x0
    max_width = right_reserved - current_left - 1
    if max_width > 0 and pos_sig.width > max_width:
        new_width = max_width
        ax_sig.set_position([pos_sig.x0, pos_sig.y0, new_width, pos_sig.height])

    # create colorbar for the top axis and capture its axes so we can align the
    # bottom legend to the same reserved area on the right
    cbar = fig.colorbar(scatter_xy, ax=ax, label="ADF amplitude normalised")

    x_min = min(xant_vis.min(), xmax_pos_vis[0], tau_pos_vis[0]) / 1e3
    x_max = max(xant_vis.max(), xmax_pos_vis[0], tau_pos_vis[0]) / 1e3
    y_min = min(yant_vis.min(), xmax_pos_vis[1], tau_pos_vis[1]) / 1e3
    y_max = max(yant_vis.max(), xmax_pos_vis[1], tau_pos_vis[1]) / 1e3

    x_pad = max(0.05 * (x_max - x_min), 0.05)
    y_pad = max(0.05 * (y_max - y_min), 0.05)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Limit the displayed trace on ax_sig to ±100 ns around the peak (center frame)
    # compute envelope on the center frame to find peak time
    env_center = np.linalg.norm(beam_trace_matrix[center_idx, 1:], axis=0)
    peak_idx = int(np.nanargmax(env_center))
    peak_time = float(beam_time_axis[peak_idx])
    window_half_ns = 100.0
    window_min = peak_time - window_half_ns
    window_max = peak_time + window_half_ns

    # build a mask for the time axis inside the window (fall back to full range if empty)
    time_mask = (beam_time_axis >= window_min) & (beam_time_axis <= window_max)
    if not np.any(time_mask):
        # fallback to full span
        xlim_min = float(beam_time_axis.min())
        xlim_max = float(beam_time_axis.max())
        amp_slice = beam_trace_matrix[:, 1:, :]
    else:
        xlim_min = float(window_min)
        xlim_max = float(window_max)
        amp_slice = beam_trace_matrix[:, 1:, time_mask]

    # set amplitude limits based on the selected time window so y-limits match visible data
    beam_amp_vmax = float(np.nanmax(np.linalg.norm(beam_trace_matrix[center_idx, 1:], axis=0)))
    beam_amp_vmin = -beam_amp_vmax
    ax_sig.set_xlim(xlim_min, xlim_max)
    ax_sig.set_ylim(beam_amp_vmin, beam_amp_vmax * 1.05)

    beam_line_x, = ax_sig.plot(beam_time_axis, beam_trace_matrix[center_idx, 1], color="tab:blue", lw=1.2, label="$\\sum$ X")
    beam_line_y, = ax_sig.plot(beam_time_axis, beam_trace_matrix[center_idx, 2], color="tab:orange", lw=1.2, label="$\\sum$ Y")
    beam_line_z, = ax_sig.plot(beam_time_axis, beam_trace_matrix[center_idx, 3], color="tab:green", lw=1.2, label="$\\sum$ Z")
    beam_line_env, = ax_sig.plot(
        beam_time_axis,
        np.linalg.norm(beam_trace_matrix[center_idx, 1:], axis=0),
        color="black",
        lw=2.0,
        alpha=0.75,
        label="|$\\sum$ E|",
    )
    # place the bottom plot legend in the reserved right whitespace (under the colorbar)
    # determine colorbar left edge in figure coords
    try:
        cbar_pos = cbar.ax.get_position()
        cbar_left = float(cbar_pos.x0)
        cbar_right = float(cbar_pos.x1)
        # tighten right figure margin so there's minimal whitespace beyond the colorbar
        fig.subplots_adjust(right=min(0.98, cbar_right + 0.01))
    except Exception:
        cbar_left = 0.82

    # compute new width so ax_sig right edge equals slightly left of cbar_left
    pos_sig = ax_sig.get_position()
    pad = 0.005
    extra_shorten = 0.05
    new_width = max(0.05, cbar_left - pos_sig.x0 - pad - extra_shorten)
    ax_sig.set_position([pos_sig.x0, pos_sig.y0, new_width, pos_sig.height])

    # place legend in figure coordinates inside the reserved area (under the colorbar)
    legend_x = cbar_left - 0.05
    legend_y = pos_sig.y0 + pos_sig.height * 0.9
    ax_sig.legend(loc="upper left", bbox_to_anchor=(legend_x, legend_y), bbox_transform=fig.transFigure, framealpha=0.9)

    pause_start = center_idx + 1
    pause_end = pause_start + pause_frames

    def update(frame_index: int):
        phi_frame = phi_values[frame_index]
        adf = adf_matrix[frame_index]
        beam_trace = beam_trace_matrix[frame_index]
        # update colors only; keep marker sizes constant
        scatter_xy.set_array(adf)

        
        # If we're in the pause region (frames during the inserted center pause), show success overlay
        if pause_frames > 0 and (pause_start - 1 <= frame_index < pause_end):
            # highlight line and show success
            line_xy.set_color("green")
            line_xy.set_linewidth(3.0)
            success_text.set_text("SUCCEED")
            success_text.set_alpha(1.0)
            phi_frame = -np.pi /2
            fig.suptitle(
                "Illustration only | "
                f"ADF sweeping | "
                f"phi offset = {0:+.2f}°",
                fontsize=16,
            )
        else:
            line_xy.set_color("blue")
            line_xy.set_linewidth(1.5)
            success_text.set_text("")
            success_text.set_alpha(0.0)

            fig.suptitle(
                "Illustration only | "
                f"ADF sweeping | "
                f"phi offset = {phi_values_deg[frame_index]:+.2f}°",
                fontsize=16,
            )
        shower_dir_frame = shower_direction_vector(theta, phi_frame)
        end_point = xmax_pos_vis + shower_dir_frame * np.linalg.norm(tau_pos_vis)

        line_xy.set_data(
            [xmax_pos_vis[0] / 1e3, end_point[0] / 1e3],
            [xmax_pos_vis[1] / 1e3, end_point[1] / 1e3],
        )

        beam_line_x.set_data(beam_time_axis, beam_trace[1])
        beam_line_y.set_data(beam_time_axis, beam_trace[2])
        beam_line_z.set_data(beam_time_axis, beam_trace[3])
        beam_line_env.set_data(beam_time_axis, np.linalg.norm(beam_trace[1:], axis=0))

        info_text.set_text(
            f"theta = {theta * R2D:.2f}°  |  phi = {phi_frame * R2D:.2f}°"
        )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=2, framealpha=0.9)
        return scatter_xy, line_xy, beam_line_x, beam_line_y, beam_line_z, beam_line_env, info_text, success_text

    anim = FuncAnimation(fig, update, frames=len(phi_values), interval=120, blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=8)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Animate the ADF antenna-layout plot while sweeping phi.")
    parser.add_argument("--event-id", type=int, default=110, help="Event id to animate.")
    parser.add_argument("--output", type=Path, default=Path("plot_slides/adf_phi_animation.gif"), help="Output GIF path.")
    parser.add_argument("--phi-span-deg", type=float, default=4.0, help="Total phi sweep span in degrees.")
    parser.add_argument("--frames", type=int, default=60, help="Number of animation frames.")
    parser.add_argument("--pause-frames", type=int, default=12, help="Number of frames to pause at true phi (center).")
    parser.add_argument("--delta-omega", type=float, default=1.0, help="ADF delta_omega parameter.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    event = load_event(args.event_id, config)
    build_animation(
        event=event,
        output_path=args.output,
        phi_span_deg=args.phi_span_deg,
        frames=args.frames,
        delta_omega=args.delta_omega,
        pause_frames=args.pause_frames,
    )
    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()

# relever legend
# plus de padding sous xmax pour place texte et legend