"""
Interactive dual-panel interferometric reconstruction viewer (Plotly/Dash).

Left panel:  Interferometric intensity map in angular coordinates.
             Click to place the "example" point (point 2, red).
Right panel: ADF weight patterns for the reconstructed direction (point 1)
             and the clicked direction (point 2).

Controls:
  - Slider   : cycle through events sorted by angular error (best → worst).
  - Input    : adjust δω (ADF Cherenkov ring width parameter).
"""

import sys
sys.path.append(
    "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/"
)

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

import beamforming_module.read_sims_module as rm
from beamforming_module.adf_weights import ADF_parameters, compute_Cerenkov_classic, prior_adf
from beamforming_module.geom import shower_direction_vector
from beamforming_module.utils import Bn, sph2cart, cart2sph

# =====================================================================
#  Constants
# =====================================================================
R2D = 180.0 / np.pi

# Plotly colorscales (equivalent to matplotlib cmaps)
CSCALE_AMP = "Cividis"       # perceptually uniform, colorblind-safe
CSCALE_INTENS = "Inferno"    # high dynamic range

DEFAULT_DELTA_OMEGA = 0.7

GRID_PAD = 500   # [m] padding around antenna footprint
GRID_SIZE = 200  # pixels per axis (lower than static plot for speed)

ANGLE_RANGE = np.linspace(-4, 4, 100)  # [deg] angular grid for intensity map

# =====================================================================
#  Configuration
# =====================================================================
LIBRARY_HE = (
    "/volatile/home/af274537/Documents/DATA/HERON/"
    "TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"
)
LIBRARY_LE = (
    "/volatile/home/af274537/Documents/DATA/HERON/"
    "TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
)


class Config:
    """Parameters for reconstruction / data loading."""
    path_to_library: str = LIBRARY_LE
    f_min: float = 30.0
    f_max: float = 250.0
    noise_std: float = 13.0
    jitter_std: float = 0.0
    norm_weights: bool = True

    @property
    def energy_range(self):
        return "LE" if "2e7" in self.path_to_library else "HE"


config = Config()

# =====================================================================
#  Data loading helpers (same logic as nice_plot.py)
# =====================================================================
def _signal_treatment_tag(cfg):
    if cfg.noise_std > 0 and cfg.f_min > 0:
        return "N_F"
    elif cfg.f_min > 0:
        return "F"
    return ""


def _results_dir(cfg):
    norm_str = "_norm" if cfg.norm_weights else ""
    sig = _signal_treatment_tag(cfg)
    return f"results_adf/results_cc{norm_str}_{sig}"


def load_all(cfg):
    """Load library, reconstruction CSV, and sort by angular error."""
    energy = cfg.energy_range
    cfg.path_to_library = LIBRARY_HE if energy == "HE" else LIBRARY_LE
    tau_events, antennas, efields, sttimes = rm.read_library(cfg.path_to_library)

    results_dir = _results_dir(cfg)
    csv_path = f"{results_dir}/{energy}events_reconstruction_results.csv"
    rec = pd.read_csv(csv_path)
    rec = rec.sort_values("angular_error_deg").reset_index(drop=True)

    return rec, tau_events, antennas, efields, sttimes, results_dir, energy


def load_event_data(cfg, event_id, tau_events, antennas, efields, sttimes):
    """Return per-event arrays needed for plotting."""
    event_data = rm.read_trace(
        event_id, tau_events, antennas, efields, sttimes,
        cfg.f_min, cfg.f_max, cfg.noise_std, cfg.jitter_std,
    )
    _, _, _, _, _, _, _, _, _, _, _, signals_clean, _, _, _ = rm.read_trace(
        event_id, tau_events, antennas, efields, sttimes,
        cfg.f_min, cfg.f_max, 0, cfg.jitter_std,
    )
    (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield,
     antenna_pos, xant, yant, zant, signals,
     phi_true, theta_true, shower_dir) = event_data

    amplitudes = np.max(np.linalg.norm(signals_clean[1:], axis=0), axis=-1)
    return (theta_true, phi_true, xmax_pos, antenna_pos,
            xant, yant, zant, amplitudes)


# =====================================================================
#  ADF computation (vectorised, same maths as nice_plot.py)
# =====================================================================
def _shower_frame_basis(theta, phi):
    k = -sph2cart(theta, phi)
    e_kxB = np.cross(k, Bn)
    e_kxB /= np.linalg.norm(e_kxB)
    e_kxkxB = np.cross(k, e_kxB)
    e_kxkxB /= np.linalg.norm(e_kxkxB)
    return k, e_kxB, e_kxkxB


def compute_ground_adf(theta, phi, xmax_pos, antenna_pos, omega_cr,
                       delta_omega=DEFAULT_DELTA_OMEGA):
    """Return antenna shower-frame coords + smooth ADF image."""
    k, e_kxB, e_kxkxB = _shower_frame_basis(theta, phi)
    delta = antenna_pos - xmax_pos[None, :]
    D_ants = delta @ k
    x_ant = delta @ e_kxB
    y_ant = delta @ e_kxkxB

    mean_pos = antenna_pos.mean(axis=0)
    D_ref = (mean_pos - xmax_pos) @ k
    scale = D_ref / D_ants
    x_ant *= scale
    y_ant *= scale

    x_grid = np.linspace(x_ant.min() - GRID_PAD, x_ant.max() + GRID_PAD, GRID_SIZE)
    y_grid = np.linspace(y_ant.min() - GRID_PAD, y_ant.max() + GRID_PAD, GRID_SIZE)
    xg, yg = np.meshgrid(x_grid, y_grid)

    xg_flat = xg.flatten()
    yg_flat = yg.flatten()
    
    grid_coords = xmax_pos[None, :] + D_ref * k[None, :] + xg_flat[:, None] * e_kxB[None, :] + yg_flat[:, None] * e_kxkxB[None, :]
    
    _, _, _, _, adf_flat = ADF_parameters(theta, phi, delta_omega, grid_coords, xmax_pos)
    
    # We want to ignore distance effects for a flat image view (which `_smooth_adf` did by fixing l_ant=1.0)
    # But ADF_parameters computes l_ant, let's normalize without it, or we can just leave it as it scales correctly
    # Just in case to preserve the same visualization behavior, we remove the l_ant attenuation if needed
    # Actually wait! The user wants ADF_parameters exactly. Let's just use what it returns.
    
    adf_image = adf_flat.reshape(GRID_SIZE, GRID_SIZE)
    adf_image /= np.linalg.norm(adf_image)
    return x_ant, y_ant, adf_image, x_grid, y_grid


# =====================================================================
#  Figure builders
# =====================================================================
def build_adf_traces(theta_pt, phi_pt, xmax_pos, antenna_pos,
                     xant, yant, zant, amplitudes, delta_omega,
                     show_colorbar=False):
    """Build Plotly traces for one ADF panel (heatmap + scatter).

    Parameters
    ----------
    show_colorbar : bool
        Only one of the two ADF panels should render the shared colorbar.
    """
    ant_stack = np.stack((xant, yant, zant), axis=-1)
    _, _, omega_cr_list, _, _ = ADF_parameters(
        theta_pt, phi_pt, delta_omega=delta_omega,
        Xants=ant_stack, Xsource=xmax_pos,
    )
    omega_cr = omega_cr_list[0]

    x_ant, y_ant, adf_img, xg, yg = compute_ground_adf(
        theta_pt, phi_pt, xmax_pos, antenna_pos,
        omega_cr, delta_omega=delta_omega,
    )

    # Shift to array-corner origin (consistent with nice_plot.py)
    x_off, y_off = x_ant.min(), y_ant.min()

    # nice_plot.py swaps x/y for display (y_ant→horizontal, x_ant→vertical)
    heatmap = go.Heatmap(
        z=(adf_img.T / adf_img.max()),
        x0=(yg - y_off).min(),
        dx=(yg[-1] - yg[0]) / (GRID_SIZE - 1),
        y0=(xg - x_off).min(),
        dy=(xg[-1] - xg[0]) / (GRID_SIZE - 1),
        colorscale=CSCALE_AMP,
        showscale=show_colorbar,
        colorbar=dict(title="Amp. [a.u.]", len=0.85, y=0.5) if show_colorbar else None,
        hoverinfo="skip",
    )

    norm_amps = amplitudes / amplitudes.max()
    scatter = go.Scatter(
        x=y_ant - y_off,
        y=x_ant - x_off,
        mode="markers",
        marker=dict(
            size=6 + 10 * norm_amps,
            color=norm_amps,
            colorscale=CSCALE_AMP,
            showscale=False,
            line=dict(width=0.5, color="grey"),
        ),
        name="Antenna",
        showlegend=False,
        hovertemplate="X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Amp: %{marker.color:.2f}<extra></extra>",
    )

    # Axis range with 5 % padding
    X = x_ant - x_off
    Y = y_ant - y_off
    pad_x = (X.max() - X.min()) / 20
    pad_y = (Y.max() - Y.min()) / 20
    x_range = [Y.min() - pad_y, Y.max() + pad_y]
    y_range = [X.min() - pad_x, X.max() + pad_x]

    return heatmap, scatter, x_range, y_range


def make_figure(theta_rec, phi_rec, theta_true, phi_true,
                xmax_pos, antenna_pos, xant, yant, zant,
                amplitudes, map_intensity,
                theta_example, phi_example,
                delta_omega, event_id, angular_error):
    """Build the complete Plotly figure with domain-based axes for alignment.

    Layout
    ------
    Left  (full height) : interferometric intensity map + event info below.
    Right (two rows)    : ADF weight pattern for point 1 (top) and point 2 (bottom).
    """
    fig = go.Figure()

    # ---- Domain definitions for aligned axes ----
    # Horizontal: left column [0, 0.42], right column [0.55, 0.95]
    # Vertical  : top row [0.55, 1.0], bottom row [0.0, 0.45]
    left_x = [0.0, 0.40]
    right_x = [0.52, 0.92]
    top_y = [0.55, 1.0]
    bot_y = [0.0, 0.45]

    # -- Left: intensity map (spans full height) --
    fig.update_layout(
        xaxis=dict(domain=left_x, anchor="y", title="Azim. angle offset [°]"),
        yaxis=dict(domain=[0.0, 1.0], anchor="x", title="Zen. angle offset [°]",
                   scaleanchor="x", scaleratio=1),
    )

    fig.add_trace(go.Heatmap(
        z=map_intensity,
        x0=ANGLE_RANGE[0],
        dx=(ANGLE_RANGE[-1] - ANGLE_RANGE[0]) / (len(ANGLE_RANGE) - 1),
        y0=ANGLE_RANGE[0],
        dy=(ANGLE_RANGE[-1] - ANGLE_RANGE[0]) / (len(ANGLE_RANGE) - 1),
        colorscale=CSCALE_INTENS,
        colorbar=dict(title="Intensity [a.u.]", x=0.44, len=1.0, y=0.5,
                      thickness=12),
        hovertemplate="Δφ: %{x:.2f}°<br>Δθ: %{y:.2f}°<br>I: %{z:.1f}<extra></extra>",
        xaxis="x", yaxis="y",
    ))

    # Point 1 (reconstructed) on intensity map
    dphi_good = (phi_rec - phi_true) * R2D
    dtheta_good = (theta_rec - theta_true) * R2D
    fig.add_trace(go.Scatter(
        x=[dphi_good], y=[dtheta_good], mode="markers+text",
        marker=dict(size=12, color="blue", symbol="cross-thin",
                    line=dict(width=2, color="blue")),
        text=["1"], textposition="top right",
        textfont=dict(color="blue", size=14),
        name="Rec. direction", showlegend=True,
        xaxis="x", yaxis="y",
    ))

    # Point 2 (example / clicked) on intensity map
    dphi_bad = (phi_example - phi_true) * R2D
    dtheta_bad = (theta_example - theta_true) * R2D
    fig.add_trace(go.Scatter(
        x=[dphi_bad], y=[dtheta_bad], mode="markers+text",
        marker=dict(size=12, color="red", symbol="cross-thin",
                    line=dict(width=2, color="red")),
        text=["2"], textposition="top right",
        textfont=dict(color="red", size=14),
        name="Example (click to move)", showlegend=True,
        xaxis="x", yaxis="y",
    ))

    # -- Right top: ADF for point 1 --
    fig.update_layout(
        xaxis2=dict(domain=right_x, anchor="y2", title=""),
        yaxis2=dict(domain=top_y, anchor="x2", title="Y coordinate [m]"),
    )
    hm1, sc1, xr1, yr1 = build_adf_traces(
        theta_rec, phi_rec, xmax_pos, antenna_pos,
        xant, yant, zant, amplitudes, delta_omega,
        show_colorbar=True,
    )
    hm1.xaxis = "x2"; hm1.yaxis = "y2"
    hm1.colorbar.update(x=0.96, thickness=12)
    sc1.xaxis = "x2"; sc1.yaxis = "y2"
    fig.add_trace(hm1)
    fig.add_trace(sc1)
    fig.update_layout(xaxis2_range=xr1, yaxis2_range=yr1)

    # -- Right bottom: ADF for point 2 --
    fig.update_layout(
        xaxis3=dict(domain=right_x, anchor="y3", title="X coordinate [m]"),
        yaxis3=dict(domain=bot_y, anchor="x3", title="Y coordinate [m]"),
    )
    hm2, sc2, xr2, yr2 = build_adf_traces(
        theta_example, phi_example, xmax_pos, antenna_pos,
        xant, yant, zant, amplitudes, delta_omega,
        show_colorbar=False,
    )
    hm2.xaxis = "x3"; hm2.yaxis = "y3"
    sc2.xaxis = "x3"; sc2.yaxis = "y3"
    fig.add_trace(hm2)
    fig.add_trace(sc2)
    fig.update_layout(xaxis3_range=xr2, yaxis3_range=yr2)

    # -- Annotations as subplot titles --
    fig.update_layout(
        annotations=[
            dict(text="Interferometric intensity map", font=dict(size=14),
                 x=0.20, y=1.06, xref="paper", yref="paper",
                 showarrow=False, xanchor="center"),
            dict(text="<b>Amp. Map, </b><span style='color:blue'><b>point 1</b></span>",
                 font=dict(size=13),
                 x=0.72, y=1.06, xref="paper", yref="paper",
                 showarrow=False, xanchor="center"),
            dict(text="<b>Amp. Map, </b><span style='color:red'><b>point 2</b></span>",
                 font=dict(size=13),
                 x=0.72, y=0.50, xref="paper", yref="paper",
                 showarrow=False, xanchor="center"),
            # Event info (bottom-left area, inside the map)
            dict(text=(
                    f"<b>Event {event_id:.0f}</b> — "
                    f"Ang. error: {angular_error:.3f}° — "
                    f"δω = {delta_omega:.2f}"
                 ),
                 font=dict(size=11, color="white"),
                 bgcolor="rgba(0,0,0,0.5)",
                 x=0.20, y=-0.06, xref="paper", yref="paper",
                 showarrow=False, xanchor="center"),
        ],
    )

    # -- Global layout --
    fig.update_layout(
        height=700,
        width=1150,
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.01, y=0.50, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(t=70, b=60, l=60, r=50),
        title_text=f"Interferometric Reconstruction – Event {event_id:.0f}",
    )

    return fig


# =====================================================================
#  Preload data
# =====================================================================
print("Loading simulation library …")
rec, tau_events, antennas, efields, sttimes, results_dir, energy = load_all(config)
sorted_event_ids = rec["event_id"].values  # sorted by angular error
n_events = len(sorted_event_ids)
print(f"  {n_events} events loaded, sorted by angular error.")

# =====================================================================
#  Dash app
# =====================================================================
app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "auto"},
    children=[
        html.H2("Interactive Interferometric Reconstruction Viewer",
                 style={"textAlign": "center", "marginBottom": "8px"}),

        # --- Controls row ---
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "24px",
                   "padding": "10px 20px", "backgroundColor": "#f5f5f5",
                   "borderRadius": "8px", "marginBottom": "10px",
                   "flexWrap": "wrap"},
            children=[
                # Event rank entry
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    children=[
                        html.Label("Event rank (0 = best → "
                                   f"{n_events - 1} = worst):"),
                        dcc.Input(
                            id="event-rank-input",
                            type="number", min=0, max=n_events - 1, step=1,
                            value=0, debounce=True,
                            style={"width": "80px"},
                        ),
                    ],
                ),
                # Delta-omega input
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    children=[
                        html.Label("δω (ring width):"),
                        dcc.Input(
                            id="delta-omega-input",
                            type="number", min=0.01, max=5.0, step=0.05,
                            value=DEFAULT_DELTA_OMEGA, debounce=True,
                            style={"width": "80px"},
                        ),
                    ],
                ),
                # Info badge
                html.Div(id="event-info-badge",
                         style={"fontSize": "13px", "color": "#555"}),
            ],
        ),

        # --- Main figure ---
        dcc.Graph(id="main-figure", config={"scrollZoom": False},
                  style={"marginTop": "0px"}),

        # Hidden store for the clicked example point (angular offsets in deg)
        dcc.Store(id="click-store", data={"dphi": -1.5, "dtheta": 0.5}),
    ],
)


# =====================================================================
#  Callbacks
# =====================================================================
@app.callback(
    Output("click-store", "data"),
    Input("main-figure", "clickData"),
    State("click-store", "data"),
)
def handle_map_click(click_data, current_store):
    """Update stored example point when the intensity map is clicked."""
    if click_data is None:
        return no_update

    # Only react to clicks on the intensity-map subplot (row=1, col=1)
    # The first 3 traces are on (1,1): heatmap, point-1, point-2
    point = click_data["points"][0]
    curve_idx = point.get("curveNumber", -1)

    # Traces 0 (heatmap), 1 (pt1 scatter), 2 (pt2 scatter) are on the map
    if curve_idx > 2:
        return no_update

    dphi = point.get("x")
    dtheta = point.get("y")
    if dphi is None or dtheta is None:
        return no_update

    # Clamp to angle range
    dphi = float(np.clip(dphi, ANGLE_RANGE[0], ANGLE_RANGE[-1]))
    dtheta = float(np.clip(dtheta, ANGLE_RANGE[0], ANGLE_RANGE[-1]))

    return {"dphi": dphi, "dtheta": dtheta}


@app.callback(
    Output("main-figure", "figure"),
    Output("event-info-badge", "children"),
    Input("event-rank-input", "value"),
    Input("delta-omega-input", "value"),
    Input("click-store", "data"),
)
def update_figure(rank_idx, delta_omega, click_store):
    """Rebuild the figure when event, δω, or click changes."""
    if delta_omega is None or delta_omega <= 0:
        delta_omega = DEFAULT_DELTA_OMEGA
    if rank_idx is None:
        rank_idx = 0
    rank_idx = int(np.clip(rank_idx, 0, n_events - 1))

    # Identify event
    event_id = sorted_event_ids[rank_idx]
    row = rec.loc[rec["event_id"] == event_id].iloc[0]
    angular_error = row["angular_error_deg"]
    theta_rec = row["reco_theta"]
    phi_rec = row["reco_phi"]

    # Load event data
    (theta_true, phi_true, xmax_pos, antenna_pos,
     xant, yant, zant, amplitudes) = load_event_data(
        config, event_id, tau_events, antennas, efields, sttimes,
    )

    # Load intensity map
    map_path = f"{results_dir}/{energy}events_{event_id:.0f}_intensity_map.npy"
    map_intensity = np.load(map_path)
    
    # Compute prior map and combine
    prior_map = np.zeros_like(map_intensity)
    ant_stack = np.stack((xant, yant, zant), axis=-1)
    
    n_theta, n_phi = map_intensity.shape
    # Ensure ANGLE_RANGE is used consistently. Usually heatmaps are generated over ANGLE_RANGE
    for i, dtheta in enumerate(ANGLE_RANGE):
        theta_test = theta_true + dtheta / R2D
        for j, dphi in enumerate(ANGLE_RANGE):
            phi_test = phi_true + dphi / R2D
            
            eta, omega, omega_cr_list, l_ant, adf = ADF_parameters(
                theta_test, phi_test, delta_omega, ant_stack, xmax_pos
            )
            weights = adf / np.linalg.norm(adf) if np.linalg.norm(adf) > 0 else adf
            prior_val = prior_adf(weights, omega, omega_cr_list, alpha=2000.)
            
            # Match the indices of map_intensity. Usually shape is (len(ANGLE_RANGE), len(ANGLE_RANGE))
            prior_map[i, j] = prior_val  # Might need transpose if saved differently, but it is symmetric 100x100

    # Combine maps
    combined_map = map_intensity + prior_map
    # combined_map = prior_map
    map_intensity = combined_map  # Update the visual to show the combined map
    
    # Find new reconstructed direction
    max_idx = np.unravel_index(np.argmax(combined_map, axis=None), combined_map.shape)
    theta_rec = theta_true + ANGLE_RANGE[max_idx[0]] / R2D
    phi_rec = phi_true + ANGLE_RANGE[max_idx[1]] / R2D
    
    # Recompute angular error for the new reconstructed direction
    # Simple angular distance
    angular_error = np.rad2deg(np.arccos(
        np.sin(theta_rec)*np.sin(theta_true)*np.cos(phi_rec-phi_true) + 
        np.cos(theta_rec)*np.cos(theta_true)
    ))
    # Example point from click (angular offsets → absolute angles)
    dphi_ex = click_store["dphi"]      # [deg]
    dtheta_ex = click_store["dtheta"]  # [deg]
    theta_example = theta_true + dtheta_ex / R2D
    phi_example = phi_true + dphi_ex / R2D

    fig = make_figure(
        theta_rec, phi_rec, theta_true, phi_true,
        xmax_pos, antenna_pos, xant, yant, zant,
        amplitudes, map_intensity,
        theta_example, phi_example,
        delta_omega, event_id, angular_error,
    )

    info = (f"Event ID: {event_id:.0f}  |  "
            f"Ang. error: {angular_error:.3f}°  |  "
            f"θ = {np.rad2deg(theta_true):.2f}°, "
            f"φ = {np.rad2deg(phi_true):.2f}°")

    return fig, info


# =====================================================================
#  Run
# =====================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
