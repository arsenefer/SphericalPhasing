"""
Dual-panel interferometric reconstruction comparison plot.

Left panel:  Interferometric intensity map in angular coordinates.
Right panel: ADF (Antenna Distribution Function) weight patterns projected
             onto the ground plane for two test directions.

Used for publication figures in radio cosmic-ray reconstruction studies.
"""
import sys
sys.path.append("/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/")  # noqa: E501
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


import beamforming_module.read_sims_module as rm
from beamforming_module.adf_weights import (
    ADF_parameters, compute_Cerenkov_classic,
)
from beamforming_module.geom import shower_direction_vector
from beamforming_module.utils import Bn, sph2cart

# ---------------------------------------------------------------------------
#  Constants & matplotlib style
# ---------------------------------------------------------------------------
R2D = 180.0 / np.pi  # rad → deg

SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14

plt.rc("font", size=LARGE_SIZE)
plt.rc("axes", titlesize=LARGE_SIZE, labelsize=LARGE_SIZE)
plt.rc("xtick", labelsize=MEDIUM_SIZE)
plt.rc("ytick", labelsize=MEDIUM_SIZE)
plt.rc("legend", fontsize=LARGE_SIZE)
plt.rc("figure", titlesize=LARGE_SIZE)

# Colormap choices for interferometric data:
#   'cividis'  – perceptually uniform, colorblind-safe (good for amplitude)
#   'inferno'  – high dynamic-range data
#   'RdBu_r'   – diverging, suited for phase maps
CMAP_AMP = "cividis"
CMAP_INTENS = "inferno"

# Default ADF shape parameter (Cherenkov ring width)
DEFAULT_DELTA_OMEGA = 0.7

# Grid padding around antenna footprint [m]
GRID_PAD = 500
GRID_SIZE = 500  # pixels per axis for smooth ADF image

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
LIBRARY_HE = (
    "/volatile/home/af274537/Documents/DATA/HERON/"
    "TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"
)
LIBRARY_LE = (
    "/volatile/home/af274537/Documents/DATA/HERON/"
    "TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
)

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf_bruitfilt'
    sampling: str = 'random'
    n_walkers: int = 200
    n_steps: int = 400
    step_size: float = 1000.
    f_min: float = 30
    f_max: float = 250
    noise_std: float = 13
    jitter_std: float = 0
    bounds: str = 'flat_sphere'  # Options: 'cubic', 'sphere', 'flat_sphere'
    r: float = 10.e3  # Radius for spherical bounds
    thickness: float = 4e3  # Thickness for flat spherical bounds
    intens_method: str = 'amplitude'  # Intensity calculation method
    norm_weights: bool = True  # Whether to normalize ADF weights
    temp: float = 5000.  # Temperature for sampling
    x_max_method: str = 'max'  # Method for finding maximum
    n_best_walkers: int = 0  # Number of best walkers to consider
    sep_walkers: bool = False  # Separate walkers flag
    burn_in: int = 0  # Number of burn-in steps to discard

    @property
    def energy_range(self):
        return 'LE' if '2e7' in self.path_to_library else 'HE'
   




# ===================================================================
#  Data I/O
# ===================================================================
def load_reconstruction_data(
    config, 
    energy: str = "LE",
    signal_treatment: str = "N_F",
    normalised: bool = True,
):
    """
    Load the simulation library and the corresponding reconstruction CSV.

    Parameters
    ----------
    energy : {"LE", "HE"}
        Energy regime (low / high).
    signal_treatment : str
        Signal processing tag used in the results subfolder name.
    normalised : bool
        Whether to use the normalised CC reconstruction results.

    Returns
    -------
    rec : pd.DataFrame
        Reconstruction results table.
    tau_events, antennas, efields, sttimes : arrays
        Raw simulation library contents.
    """
    config.path_to_library = LIBRARY_HE if energy == "HE" else LIBRARY_LE
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)

    norm_tag = "_norm" if normalised else ""
    csv_path = (
        f"results_adf/ADF_Fixed_norm_N_F_both/"
        f"{energy}events_reconstruction_results.csv"
    )
    rec = pd.read_csv(csv_path)
    return rec, tau_events, antennas, efields, sttimes


def load_event(config, event_id, tau_events, antennas, efields, sttimes, noise_std=13):
    """
    Extract a single event's trace data (with and without noise).

    Parameters
    ----------
    event_id : int
        Index into the event library.
    noise_std : float
        Noise standard deviation [µV/m] for the noisy copy.

    Returns
    -------
    event_data : tuple
        (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield,
         antenna_pos, xant, yant, zant, signals, phi, theta, shower_dir)
    signals_clean : ndarray
        Noise-free signal copy.
    """
    event_data = rm.read_trace(
        event_id, tau_events, antennas, efields, sttimes,
        config.f_min, config.f_max, noise_std, config.jitter_std,
    )
    # Noise-free copy (only need the signals array)
    _, _, _, _, _, _, _, _, _, _, _, signals_clean, _, _, _ = rm.read_trace(
        event_id, tau_events, antennas, efields, sttimes,
        config.f_min, config.f_max, 0, config.jitter_std,
    )
    return event_data, signals_clean


# ===================================================================
#  ADF computation on a smooth grid
# ===================================================================
def _shower_frame_basis(theta, phi):
    """
    Build the (k×B, k×k×B, k) orthonormal basis for the shower frame.

    Returns
    -------
    k, e_kxB, e_kxkxB : ndarray, shape (3,)
    """
    k = -sph2cart(theta, phi)  # unit vector *toward* source
    e_kxB = np.cross(k, Bn)
    e_kxB /= np.linalg.norm(e_kxB)
    e_kxkxB = np.cross(k, e_kxB)
    e_kxkxB /= np.linalg.norm(e_kxkxB)
    return k, e_kxB, e_kxkxB


def _smooth_adf(theta, phi, omega, omega_cr, eta, l_ant,
                delta_omega=DEFAULT_DELTA_OMEGA):
    """
    Evaluate the analytic ADF (amplitude distribution function).

    Parameters
    ----------
    theta, phi : float
        Shower direction angles [rad].
    omega : ndarray
        Opening angle from shower axis [rad].
    omega_cr : float
        Cherenkov angle [rad].
    eta : ndarray
        Azimuthal angle in the shower plane [rad].
    l_ant : float or ndarray
        Distance from X_max to each evaluation point [m].
    delta_omega : float
        Width parameter of the Cherenkov ring.

    Returns
    -------
    adf : ndarray  (same shape as *omega*)
    """
    K = shower_direction_vector(theta, phi)

    # Geomagnetic asymmetry coefficient
    asym_coeff = -0.48
    sin2_alpha = 1.0 - np.dot(K, Bn) ** 2
    asym = asym_coeff / np.sqrt(sin2_alpha)

    # Cherenkov ring profile (Lorentzian-like)
    ring_ratio = (np.tan(omega) / np.tan(omega_cr)) ** 2 - 1.0
    adf = (1.0 / l_ant) / (1.0 + 4.0 * (ring_ratio / delta_omega) ** 2)
    adf *= 1.0 + asym * np.cos(eta)
    return adf


def compute_ground_adf(theta, phi, xmax_pos, antenna_pos, omega_cr,
                       delta_omega=DEFAULT_DELTA_OMEGA):
    """
    Compute the ADF weight map projected onto the antenna ground plane.

    The antenna positions are projected into the shower-frame transverse
    coordinates (k×B, k×k×B) and rescaled to a common reference distance
    (the mean array distance along the shower axis).

    Parameters
    ----------
    theta, phi : float
        Test shower direction [rad].
    xmax_pos : ndarray, shape (3,)
        X_max position [m].
    antenna_pos : ndarray, shape (N, 3)
        Antenna positions [m].
    omega_cr : float
        Cherenkov angle [rad].
    delta_omega : float
        ADF width parameter.

    Returns
    -------
    x_ant, y_ant : ndarray, shape (N,)
        Antenna transverse coordinates in shower frame [m].
    adf_image : ndarray, shape (GRID_SIZE, GRID_SIZE)
        Normalised smooth ADF pattern.
    x_grid, y_grid : ndarray, shape (GRID_SIZE,)
        Coordinate axes for `adf_image`.
    D_ref : float
        Mean distance along the shower axis (used to compute opening angle).
    """
    k, e_kxB, e_kxkxB = _shower_frame_basis(theta, phi)

    delta = antenna_pos - xmax_pos[None, :]       # (N, 3)
    D_ants = delta @ k                             # distance along shower axis
    x_ant = delta @ e_kxB                          # transverse coord 1
    y_ant = delta @ e_kxkxB                        # transverse coord 2

    # Rescale to common reference distance (mean array plane)
    mean_pos = antenna_pos.mean(axis=0)
    D_ref = (mean_pos - xmax_pos) @ k
    scale = D_ref / D_ants                         # (N,)
    x_ant *= scale
    y_ant *= scale

    # Build smooth evaluation grid
    x_grid = np.linspace(x_ant.min() - GRID_PAD, x_ant.max() + GRID_PAD, GRID_SIZE)
    y_grid = np.linspace(y_ant.min() - GRID_PAD, y_ant.max() + GRID_PAD, GRID_SIZE)
    xg, yg = np.meshgrid(x_grid, y_grid)

    r_grid = np.sqrt(xg ** 2 + yg ** 2)
    omega_grid = np.arctan2(r_grid, D_ref)
    eta_grid = np.arctan2(yg, xg)

    adf_image = _smooth_adf(theta, phi, omega_grid, omega_cr, eta_grid, 1.0,
                            delta_omega=delta_omega)
    adf_image /= np.linalg.norm(adf_image)

    return x_ant, y_ant, adf_image, x_grid, y_grid, D_ref


# ===================================================================
#  Main plotting routine
# ===================================================================
def plot_interferometric_comparison(
    theta_rec, phi_rec, theta_true, phi_true,
    xmax_pos, antenna_pos, xant, yant, zant,
    amplitudes, map_intensity,
    save_path=None,
    delta_omega=DEFAULT_DELTA_OMEGA,
):
    """
    Create the dual-panel interferometric reconstruction figure.

    Left panel
        2-D intensity map (angular coordinates centred on the true direction).
    Right panel
        ADF weight pattern for the reconstructed direction (point 1, blue)
        and an offset test direction (point 2, red).

    Parameters
    ----------
    theta_rec, phi_rec : float
        Reconstructed shower direction [rad].
    theta_true, phi_true : float
        True shower direction [rad].
    xmax_pos : ndarray (3,)
        X_max position [m].
    antenna_pos : ndarray (N, 3)
        Antenna positions [m].
    xant, yant, zant : ndarray (N,)
        Individual antenna coordinate arrays [m].
    amplitudes : ndarray (N,)
        Per-antenna signal amplitudes [a.u.].
    map_intensity : ndarray (M, M)
        Pre-computed interferometric intensity map.
    save_path : str or None
        If given, save the figure to this path (PDF recommended).
    delta_omega : float
        ADF shape parameter.
    """
    # -- Angular grid for the intensity map (offsets from true direction) ------
    angle_range = np.linspace(-4, 4, map_intensity.shape[0])

    # -- Define the two test directions ----------------------------------------
    theta_good, phi_good = theta_rec, phi_rec                     # point 1
    theta_bad, phi_bad = theta_true - 1.5 / R2D, phi_true + 0.5 / R2D  # point 2

    # -- Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(9, 5))
    subfig_l, subfig_r = fig.subfigures(1, 2, wspace=0.25, width_ratios=[1, 1])

    # Left: intensity map + legend
    gs_left = subfig_l.add_gridspec(2, 1, height_ratios=[1, 0.07])
    ax_map = subfig_l.add_subplot(gs_left[0])
    ax_legL = subfig_l.add_subplot(gs_left[1])

    # Right: two ADF panels + shared legend
    gs_right = subfig_r.add_gridspec(3, 1, height_ratios=[1, 1, 0.14], hspace=0.35)
    ax_adf1 = subfig_r.add_subplot(gs_right[0])
    ax_adf2 = subfig_r.add_subplot(gs_right[1])
    ax_legR = subfig_r.add_subplot(gs_right[2])

    # ==================================================================
    #  LEFT PANEL – interferometric intensity map
    # ==================================================================
    im = ax_map.imshow(
        map_intensity/(13*np.sqrt(3)),  # Scale to SNR units (assuming noise_std=13 µV/m)
        extent=[angle_range.min(), angle_range.max(),
                angle_range.min(), angle_range.max()],
        aspect="auto", origin="lower", cmap=CMAP_INTENS,
    )

    # Mark the two test directions
    dphi_good = (phi_good - phi_true) * R2D
    dtheta_good = (theta_good - theta_true) * R2D
    dphi_bad = (phi_bad - phi_true) * R2D
    dtheta_bad = (theta_bad - theta_true) * R2D

    ax_map.scatter(dphi_good, dtheta_good, color="blue",
                   marker="3", s=100, label="1: True direction")
    ax_map.text(dphi_good, dtheta_good, "1 ", color="blue",
                va="bottom", ha="right", fontweight="bold")

    ax_map.scatter(dphi_bad, dtheta_bad, color="darkslategrey",
                   marker="3", s=100, zorder=1, label="2: Bad direction")
    ax_map.text(dphi_bad, dtheta_bad, "2 ", color="darkslategrey",
                va="bottom", ha="right", fontweight="bold")

    ax_map.set_xlabel(r"Azim. angle offset [$\degree$]")
    ax_map.set_ylabel(r"Zenith angle offset [$\degree$]")

    ax_legL.axis("off")
    ax_map.legend(*ax_map.get_legend_handles_labels())

    subfig_l.colorbar(im, ax=ax_map, label="Interferometric amp. [SNR]",
                      fraction=0.05, pad=0.04)

    # ==================================================================
    #  RIGHT PANELS – ADF weight patterns
    # ==================================================================
    ant_stack = np.stack((xant, yant, zant), axis=-1)

    # Helper to draw one ADF panel
    def _draw_adf_panel(ax, theta_pt, phi_pt, is_top=True):
        """Fill a single ADF axes; return the image mappable."""
        # ADF parameters at antenna positions
        eta, omega, omega_cr, l_ant, adf = ADF_parameters(
            theta_pt, phi_pt, delta_omega=delta_omega,
            Xants=ant_stack, Xsource=xmax_pos,
        )

        # Smooth ADF on a grid
        x_ant, y_ant, adf_img, xg, yg, _ = compute_ground_adf(
            theta_pt, phi_pt, xmax_pos, antenna_pos,
            omega_cr[0], delta_omega=delta_omega,
        )

        # Shift coordinates so the origin is at the array corner (cleaner axes)
        x_off, y_off = x_ant.min(), y_ant.min()

        norm_amps = amplitudes / amplitudes.max()
        amp_snr = amplitudes / (13*np.sqrt(3))

        # Background: smooth ADF pattern
        mappable = ax.imshow(
            adf_img.T / adf_img.max()*amp_snr.max(),
            extent=[(yg - y_off).min(), (yg - y_off).max(),
                    (xg - x_off).min(), (xg - x_off).max()],
            aspect="auto", origin="lower", cmap=CMAP_AMP,
        )

        # Foreground: antenna scatter (size ∝ amplitude)
        ax.scatter(
            y_ant - y_off, x_ant - x_off,
            c=amp_snr, s=20+80*norm_amps,
            edgecolors="grey", cmap=CMAP_AMP, zorder=3,
        )

        # Axis limits with 5 % padding
        X = x_ant - x_off
        Y = y_ant - y_off
        pad_x = (X.max() - X.min()) / 20
        pad_y = (Y.max() - Y.min()) / 20
        ax.set_ylim(X.min() - pad_x, X.max() + pad_x)
        ax.set_xlim(Y.min() - pad_y, Y.max() + pad_y)

        if is_top:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("kxB coordinate [m]")

        return mappable

    # --- Point 1 (reconstructed direction) --------------------------------
    pm = _draw_adf_panel(ax_adf1, theta_good, phi_good, is_top=True)
    ax_adf1.set_title("Amplitude Map,      ",
                      fontdict={"weight": "bold", "size": SMALL_SIZE}, loc="center")
    ax_adf1.annotate("point 1", xy=(0.66, 1.037), xycoords="axes fraction",
                     xytext=(3, 0), textcoords="offset points",
                     fontsize=SMALL_SIZE, fontweight="bold", color="blue",
                     ha="left", va="bottom")

    # --- Point 2 (offset test direction) ----------------------------------
    pm = _draw_adf_panel(ax_adf2, theta_bad, phi_bad, is_top=False)
    ax_adf2.set_title("Amplitude Map,      ",
                      fontdict={"weight": "bold", "size": SMALL_SIZE}, loc="center")
    ax_adf2.annotate("point 2", xy=(0.66, 1.037), xycoords="axes fraction",
                     xytext=(3, 0), textcoords="offset points",
                     fontsize=SMALL_SIZE, fontweight="bold", color="darkslategrey",
                     ha="left", va="bottom")

    # Shared legend for the antenna markers
    ax_adf2.scatter([], [], c="none", s=30, edgecolors="grey",
                    label="Antenna location")
    ax_legR.axis("off")
    # ax_legR.legend(*ax_adf2.get_legend_handles_labels(),
    #                loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    subfig_r.supylabel("kxkxB coordinate [m]", x=-0.08, y=0.6, fontsize=LARGE_SIZE)
    subfig_r.colorbar(pm, ax=[ax_adf1, ax_adf2],
                      label="Signal amplitude [SNR]",
                      fraction=0.05, pad=0.04)

    # -- Save ---------------------------------------------------------------
    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    return fig
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def main():
    config = Config()
    energy = config.energy_range
    if config.noise_std > 0 and config.f_min > 0:
        signal_treatment = f"N_F"
    elif config.f_min > 0:
        signal_treatment = f"F"
    else:
        signal_treatment = ""
        
    norm = config.norm_weights
    norm_str = '_norm' if norm else ""
    # Load data for a single event (with noise)
    rec, tau_events, antennas, efields, sttimes = load_reconstruction_data(
        config, energy=energy, signal_treatment=signal_treatment, normalised=norm,
    )
    event_id = rec.sort_values('angular_error_deg').event_id.values[1]
    event_data, signals_clean = load_event(
        config, event_id, tau_events, antennas, efields, sttimes, noise_std=config.noise_std,
    )

    # Unpack the 15-element tuple from read_trace:
    # (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield,
    #  antenna_pos, xant, yant, zant, signals, phi, theta, shower_dir)
    (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield,
     antenna_pos, xant, yant, zant, signals,
     phi_true, theta_true, shower_dir) = event_data

    # Compute per-antenna peak amplitude from the E-field norm
    # signals shape: (4, N_ant, N_samples) — row 0 is time, rows 1-3 are Ex/Ey/Ez
    amplitudes = np.max(np.linalg.norm(signals_clean[1:], axis=0), axis=-1)

    # For demonstration: create a dummy interferometric intensity map
    angle_grid = np.linspace(-4, 4, 100)
    map_intensity = np.load(f'results_adf/ADF_Fixed_norm_N_F_both/{energy}events_{event_id:.0f}_intensity_map.npy')
    angle_range = np.linspace(-4, 4, map_intensity.shape[0])
    thetas = (angle_range/R2D+theta_true)[:, None] * np.ones((1, map_intensity.shape[1]))  # broadcast to 2D
    prior_montain = sigmoid(2 * ((thetas - np.pi/2) * R2D + 0.5))

    map_intensity += prior_montain*100
    theta_rec, phi_rec = rec.loc[rec.event_id == event_id, ['reco_theta', 'reco_phi']].values[0]


    # Plot the comparison figure
    fig = plot_interferometric_comparison(
        theta_rec=theta_rec,
        phi_rec=phi_rec,
        theta_true=theta_true,
        phi_true=phi_true,
        xmax_pos=xmax_pos,
        antenna_pos=antenna_pos,
        xant=xant,
        yant=yant,
        zant=zant,
        amplitudes=amplitudes,
        map_intensity=map_intensity,
        save_path="plot_slides/cc_recons_adf_comparison_prior",
        delta_omega=DEFAULT_DELTA_OMEGA,
    )
    plt.show()

if __name__ == "__main__":
    main()