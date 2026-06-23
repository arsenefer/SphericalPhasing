#!/usr/bin/env python
"""
Script to compute angular errors for reconstructed directions across all events.

For each event in the reconstruction directory:
1. Load traces and antenna information
2. Load reconstruction samples (walker positions and intensities)
3. Compute main direction from weighted PCA of walker positions
4. Compute layout direction from antenna geometry
5. Calculate angular errors relative to true shower direction
6. Save results to CSV and create comparison histogram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, '/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')

import beamforming_module.read_sims_module as rm
from beamforming_module.sampler_init import flat_sphere_bound_function
from beamforming_module.utils import Bn, cart2sph, sph2cart

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
# Helper Functions

def load_library(path_to_library):
    """Load the Tau library containing simulated events."""
    print("Loading Tau library...")
    tau_events, antennas, efields, sttimes = rm.read_library(path_to_library)
    n_events = len(tau_events)
    print(f"Library contains {n_events} events")
    return tau_events, antennas, efields, sttimes, n_events


def load_reconstruction(event_i, recons_dir):
    """Load reconstruction samples for a specific event."""
    recons_file = os.path.join(recons_dir, f'event{event_i}.npz')
    if not os.path.exists(recons_file):
        return None
    recons_samples_npz = np.load(recons_file)
    return {
        'walker_positions': recons_samples_npz['walker_positions'],
        'walker_intensities': recons_samples_npz['beamformed_intensity_array']
    }


def select_good_samples(walker_positions, walker_intensities, quantile=0.98):
    """Select high-intensity samples based on quantile threshold."""
    intensity_threshold = np.quantile(walker_intensities.flatten(), quantile)
    good_samples_mask = walker_intensities.flatten() >= intensity_threshold
    good_positions = walker_positions.reshape(-1, 3)[good_samples_mask]
    good_intensities = walker_intensities.flatten()[good_samples_mask]
    return good_positions, good_intensities


def compute_direction_from_pca(good_positions, good_intensities):
    """Compute main direction from weighted PCA of walker positions.
    Returns (main_direction, eigenvalues)."""
    if len(good_positions) < 3:
        return None, None
    
    w = np.asarray(good_intensities, dtype=float)
    w = w / w.sum()
    
    center = np.average(good_positions, axis=0, weights=w)
    Xc = good_positions - center
    
    cov = (Xc * w[:, None]).T @ Xc
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    main_direction = eigvecs[:, np.argmax(eigvals)]
    main_direction = main_direction / np.linalg.norm(main_direction)
    return main_direction, eigvals


def compute_layout_direction(signals, antenna_pos, xmax_pos):
    """Compute layout direction from antenna amplitudes."""
    amps = np.max(np.linalg.norm(signals[1:], axis=0), axis=-1)
    layout_dir = np.average(antenna_pos, axis=0, weights=amps) - xmax_pos
    layout_dir = layout_dir / np.linalg.norm(layout_dir)
    return layout_dir


def compute_eigenvalue_ratio(eigvals):
    """Compute ratio of largest eigenvalue to mean of other two."""
    if eigvals is None or len(eigvals) != 3:
        return None
    
    sorted_eigvals = np.sort(eigvals)
    largest = sorted_eigvals[-1]
    mean_others = np.max(sorted_eigvals[:-1])
    
    if mean_others == 0:
        return None
    
    return largest / mean_others

def compute_angular_errors(main_direction, layout_dir, shower_dir):
    """Compute angular errors between directions."""
    shower_dir_normalized = shower_dir / np.linalg.norm(shower_dir)
    
    # Angular error using dot product: arccos(|dot product|)
    # We use absolute value to handle 180-degree ambiguity
    error_recons = np.arccos(np.clip(np.abs(np.dot(main_direction, shower_dir_normalized)), 0, 1))
    error_layout = np.arccos(np.clip(np.abs(np.dot(layout_dir, main_direction)), 0, 1))
    
    return np.degrees(error_recons), np.degrees(error_layout)


def process_event(event_i, tau_events, antennas, efields, sttimes, recons_dir, results, params, plot_flag=False):
    """Process a single event and store angular error results."""
    # Load reconstruction
    recons_data = load_reconstruction(event_i, recons_dir)
    if recons_data is None:
        return False
    
    # Read event traces
    (_azim, _zen, _en_nu, _en_tau, _tau_pos, _xmax_pos, 
        _event_efield, _antenna_pos, _xant, _yant, _zant, 
        _signals, _phi, _theta, _shower_dir) = rm.read_trace(
        event_i, tau_events, antennas, efields, sttimes, 
        params['F_MIN'], params['F_MAX'], params['NOISE_STD'], params['JITTER_STD']
    )
    
    # Select good samples
    good_positions, good_intensities = select_good_samples(
        recons_data['walker_positions'], recons_data['walker_intensities']
    )
    
    if len(good_positions) < 3:
        return False
    
    # Compute directions
    main_direction, eigvals = compute_direction_from_pca(good_positions, good_intensities)
    main_direction = -main_direction if np.dot(main_direction, _shower_dir) < 0 else main_direction

    if main_direction is None:
        return False
    
    layout_dir = compute_layout_direction(_signals, _antenna_pos, _xmax_pos)
    
    # Compute angular errors
    error_recons, error_layout = compute_angular_errors(main_direction, layout_dir, _shower_dir)
    
    # Compute eigenvalue ratio score
    eigval_ratio = compute_eigenvalue_ratio(eigvals)
    
    layout_dir_zenith, layout_dir_azimuth = cart2sph(*(-layout_dir))
    recons_zenith, recons_azimuth = cart2sph(*(-main_direction))
    true_zenith, true_azimuth = cart2sph(*(-_shower_dir))
    # Plot if requested
    if plot_flag:
        max_eigval = np.sqrt(np.max(eigvals))
        n_points = 10
        t_values = np.linspace(-max_eigval, max_eigval, n_points)
        center_point = good_positions[np.argmax(good_intensities)]
        points_along_direction = center_point[np.newaxis, :] + main_direction[np.newaxis, :] * t_values[:, np.newaxis]
        
        plot_event_reconstruction(good_positions, good_intensities, points_along_direction,
                                    _xmax_pos, _shower_dir, layout_dir, main_direction, event_i, score=eigval_ratio)
    
    angle_true_layout = np.degrees(
        np.arccos(np.dot(layout_dir, _shower_dir)))
    results['event_id'].append(event_i)
    results['true_azim'].append(np.degrees(true_azimuth))
    results['true_zenith'].append(np.degrees(true_zenith))
    results['energy_nu'].append(_en_nu)
    results['energy_tau'].append(_en_tau)
    results['angular_error_true'].append(error_recons)
    results['angular_error_layout'].append(error_layout)
    results['eigenvalue_ratio'].append(eigval_ratio)    
    results['recons_azim'].append(np.degrees(recons_azimuth))
    results['recons_zenith'].append(np.degrees(recons_zenith))
    results['layout_azim'].append(np.degrees(layout_dir_azimuth))
    results['layout_zenith'].append(np.degrees(layout_dir_zenith))
    results['distance_to_line'] = distance_point_to_line(_xmax_pos, main_direction, _antenna_pos)
    results['angle_true_layout'].append(angle_true_layout)
    return True
    


def save_results_to_csv(results, save_path, sampling, n_walkers, n_steps):
    """Save results DataFrame to CSV file."""
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_path, f'angular_errors_{sampling}_{n_walkers}w_{n_steps}s.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    return df


def print_statistics(df):
    """Print angular error statistics."""
    print("\n=== Angular Error Statistics ===")
    print(f"Walker-based reconstruction:")
    print(f"  Mean error: {df['angular_error_true'].mean():.2f}°")
    print(f"  Median error: {df['angular_error_true'].median():.2f}°")
    print(f"  Std dev: {df['angular_error_true'].std():.2f}°")
    print(f"  Min-Max: [{df['angular_error_true'].min():.2f}°, {df['angular_error_true'].max():.2f}°]")

    print(f"\nLayout-based direction:")
    print(f"  Mean error: {df['angular_error_layout'].mean():.2f}°")
    print(f"  Median error: {df['angular_error_layout'].median():.2f}°")
    print(f"  Std dev: {df['angular_error_layout'].std():.2f}°")
    print(f"  Min-Max: [{df['angular_error_layout'].min():.2f}°, {df['angular_error_layout'].max():.2f}°]")


def plot_event_reconstruction(good_positions, good_intensities, points_along_direction, 
                               xmax_pos, shower_dir, layout_dir, recons_direction, event_i, score=None):
    """Plot event-wise reconstruction visualization (4 subplots)."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: X-Y view
    ax[0, 0].scatter(1e-3*good_positions[:, 0], 1e-3*good_positions[:, 1], 
                     c=good_intensities, s=5, alpha=0.5, cmap='jet')
    ax[0, 0].scatter(1e-3*points_along_direction[:,0], 1e-3*points_along_direction[:,1], 
                     c='black', s=50, marker='x', label='Points along Recons Direction')
    ax[0, 0].set_xlabel("X [km]")
    ax[0, 0].set_ylabel("Y [km]")
    ax[0, 0].set_title(f"Walker Positions (Top View) - Event {event_i}")
    ax[0, 0].set_aspect('equal', adjustable='datalim')
    ax[0, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[1], 5.*shower_dir[0], 5.*shower_dir[1], 
                   color='red', head_width=.2, head_length=.2)
    ax[0, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[1], 5.*layout_dir[0], 5.*layout_dir[1], 
                   color='green', head_width=.2, head_length=.2)
    ax[0, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[1], 5.*recons_direction[0], 5.*recons_direction[1], 
                   color='blue', head_width=.2, head_length=.2)
    ax[0, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[1], -5.*recons_direction[0], -5.*recons_direction[1], 
                   color='blue', head_width=.2, head_length=.2)

    # Bottom-left: X-Z view
    ax[1, 0].scatter(1e-3*good_positions[:, 0], 1e-3*good_positions[:, 2], 
                     c=good_intensities, s=5, alpha=0.5, cmap='jet')
    ax[1, 0].scatter(1e-3*points_along_direction[:,0], 1e-3*points_along_direction[:,2], 
                     c='black', s=50, marker='x', label='Points along Recons Direction')
    ax[1, 0].set_xlabel("X [km]")
    ax[1, 0].set_ylabel("Z [km]")
    ax[1, 0].set_title("Walker Positions (Side View)")
    ax[1, 0].set_aspect('equal', adjustable='datalim')
    ax[1, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[2], 5.*shower_dir[0], 5.*shower_dir[2], 
                   color='red', head_width=.2, head_length=.2)
    ax[1, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[2], 5.*layout_dir[0], 5.*layout_dir[2], 
                   color='green', head_width=.2, head_length=.2)
    ax[1, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[2], 5.*recons_direction[0], 5.*recons_direction[2], 
                   color='blue', head_width=.2, head_length=.2)
    ax[1, 0].arrow(1e-3*xmax_pos[0], 1e-3*xmax_pos[2], -5.*recons_direction[0], -5.*recons_direction[2], 
                   color='blue', head_width=.2, head_length=.2)

    # Bottom-right: Y-Z view with colorbar and legend
    sc = ax[1, 1].scatter(1e-3*good_positions[:, 1], 1e-3*good_positions[:, 2], 
                          c=good_intensities, s=5, alpha=0.5, cmap='jet')
    sc1 = ax[1, 1].scatter(1e-3*points_along_direction[:,1], 1e-3*points_along_direction[:,2], 
                           c='black', s=50, marker='x', label='Points along Recons Direction')
    ax[1, 1].set_xlabel("Y [km]")
    ax[1, 1].set_ylabel("Z [km]")
    ax[1, 1].set_title("Walker Positions (Y-Z View)")
    ax[1, 1].set_aspect('equal', adjustable='datalim')
    a1 = ax[1, 1].arrow(1e-3*xmax_pos[1], 1e-3*xmax_pos[2], 5.*shower_dir[1], 5.*shower_dir[2], 
                        color='red', head_width=.2, head_length=.2)
    a2 = ax[1, 1].arrow(1e-3*xmax_pos[1], 1e-3*xmax_pos[2], 5.*layout_dir[1], 5.*layout_dir[2], 
                        color='green', head_width=.2, head_length=.2)
    a3 = ax[1, 1].arrow(1e-3*xmax_pos[1], 1e-3*xmax_pos[2], 5.*recons_direction[1], 5.*recons_direction[2], 
                        color='blue', head_width=.2, head_length=.2)
    ax[1, 1].arrow(1e-3*xmax_pos[1], 1e-3*xmax_pos[2], -5.*recons_direction[1], -5.*recons_direction[2], 
                   color='blue', head_width=.2, head_length=.2)
    
    # Top-right: legend and colorbar
    plt.tight_layout()
    ax[0, 1].axis('off')
    ax[0, 1].legend(handles=[a1, a2, a3, sc1],
                    labels=['True Direction', 'Layout Direction', 'Recons Direction', 'Points along Recons Direction'],
                    loc='center', fontsize=10)
    cb = plt.colorbar(sc, ax=ax[0, 1], label='Beamformed Intensity (a.u.)', location='left')
    cb.solids.set_alpha(1)
    if score is not None:
        plt.suptitle(f"Event {event_i} - Eigenvalue Ratio Score: {score:.2f}", fontsize=16)
    plt.show()

def distance_point_to_line(x0, v, x):
    """Calculate distance from point x to line defined by point x0 and direction v."""
    v_norm = v / np.linalg.norm(v)
    return np.linalg.norm(np.cross(x - x0, v_norm))

def plot_results(df, save_path, sampling, n_walkers, n_steps):
    """Create and save four separate comparison figures with clear legends."""
    median_true_error = df['angular_error_true'].median()
    median_layout_error = df['angular_error_layout'].median()
    quality = df['angle_true_layout'] < 10

    # Figure 1: histogram comparison
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    max_error = max(df['angular_error_true'].max(), df['angular_error_layout'].max())
    bins = np.linspace(0, min(5, max_error), 30)
    ax1.hist(
        df['angular_error_true'],
        bins=bins,
        alpha=0.65,
        color='blue',
        edgecolor='black',
        label=f'Reconstruction error median = {median_true_error:.2f}°',
    )
    ax1.hist(
        df['angular_error_layout'],
        bins=bins,
        alpha=0.65,
        color='red',
        edgecolor='black',
        label=f'Angle to footprint dir. = {median_layout_error:.2f}°',
    )
    ax1.set_xlabel('Angular distance [°]')
    ax1.set_ylabel('Number of events')
    ax1.set_title('Angular Distance Distribution')
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=True)
    fig1.tight_layout()
    hist_path = os.path.join(save_path, f'angular_errors_hist_{sampling}_{n_walkers}w_{n_steps}s.png')
    fig1.savefig(hist_path, dpi=150, bbox_inches='tight')

    # Figure 2: angular error vs footprint angle
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(
        df.loc[quality, 'angle_true_layout'],
        df.loc[quality, 'angular_error_true'],
        alpha=0.75,
        color='blue',
        edgecolor='black',
        label='Reconstruction error',
    )
    ax2.scatter(
        df.loc[quality, 'angle_true_layout'],
        df.loc[quality, 'angular_error_layout'],
        alpha=0.75,
        color='red',
        edgecolor='black',
        label='Angle to footprint dir.',
    )
    max_scatter = max(5, float(df.loc[quality, 'angle_true_layout'].max()))
    ax2.plot([0, max_scatter], [0, max_scatter], color='gray', linestyle='--', lw=1, label='y = x')
    ax2.set_xlabel('$\\langle$ Shower direction; footprint direction $\\rangle$ [°]')
    ax2.set_ylabel('Angular distance [°]')
    ax2.set_title('Angular Error vs Footprint Angle')
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=True)
    fig2.tight_layout()
    scatter_path = os.path.join(save_path, f'angular_error_vs_footprint_{sampling}_{n_walkers}w_{n_steps}s.png')
    fig2.savefig(scatter_path, dpi=150, bbox_inches='tight')

    # Figure 3: zenith comparison
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    zenith_footprint_dist = df.loc[quality, 'layout_zenith'] - df.loc[quality, 'true_zenith']
    zenith_error = df.loc[quality, 'recons_zenith'] - df.loc[quality, 'true_zenith']
    ax3.scatter(
        zenith_footprint_dist,
        zenith_error,
        alpha=0.75,
        color='blue',
        edgecolor='black',
        label='Events',
    )
    if len(zenith_footprint_dist) > 0:
        lower = float(min(np.min(zenith_footprint_dist), np.min(zenith_error)))
        upper = float(max(np.max(zenith_footprint_dist), np.max(zenith_error)))
        ax3.plot([lower, upper], [lower, upper], color='gray', linestyle='--', lw=1, label='y = x')
    ax3.set_xlabel('$\\theta_{\\rm footprint}$ - $\\theta_{\\rm true}$ [°]')
    ax3.set_ylabel('$\\theta_{\\rm rec}$ - $\\theta_{\\rm true}$ [°]')
    ax3.set_title('Zenith Comparison')
    ax3.grid(alpha=0.3)
    ax3.legend(frameon=True)
    fig3.tight_layout()
    zen_path = os.path.join(save_path, f'zenith_comparison_{sampling}_{n_walkers}w_{n_steps}s.png')
    fig3.savefig(zen_path, dpi=150, bbox_inches='tight')

    # Figure 4: azimuth comparison
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    azimuth_footprint_dist = df.loc[quality, 'layout_azim'] - df.loc[quality, 'true_azim']
    azimuth_error = df.loc[quality, 'recons_azim'] - df.loc[quality, 'true_azim']
    ax4.scatter(
        azimuth_footprint_dist,
        azimuth_error,
        alpha=0.75,
        color='blue',
        edgecolor='black',
        label='Events',
    )
    if len(azimuth_footprint_dist) > 0:
        lower = float(min(np.min(azimuth_footprint_dist), np.min(azimuth_error)))
        upper = float(max(np.max(azimuth_footprint_dist), np.max(azimuth_error)))
        ax4.plot([lower, upper], [lower, upper], color='gray', linestyle='--', lw=1, label='y = x')
    ax4.set_xlabel('$\\phi_{\\rm footprint}$ - $\\phi_{\\rm true}$ [°]')
    ax4.set_ylabel('$\\phi_{\\rm rec}$ - $\\phi_{\\rm true}$ [°]')
    ax4.set_title('Azimuth Comparison')
    ax4.grid(alpha=0.3)
    ax4.legend(frameon=True)
    fig4.tight_layout()
    az_path = os.path.join(save_path, f'azimuth_comparison_{sampling}_{n_walkers}w_{n_steps}s.png')
    fig4.savefig(az_path, dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to:")
    print(f"  {hist_path}")
    print(f"  {scatter_path}")
    print(f"  {zen_path}")
    print(f"  {az_path}")
    plt.show()


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute angular errors for reconstructed directions across all events.'
    )
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot comparison histogram if True')
    args = parser.parse_args()

    # Configuration
    PATH_TO_LIBRARY = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    SAVE_PATH = './results'

    SAMPLING = "random"
    N_WALKERS = 200
    N_STEPS = 600
    STEP_SIZE = 1000.
    BOUNDS = "flat_sphere"
    INTENS_METHOD = "amplitude"
    TEMP = 5000.

    F_MIN, F_MAX, NOISE_STD, JITTER_STD = 50, 200, 13, 0

    # Load library
    tau_events, antennas, efields, sttimes, n_events = load_library(PATH_TO_LIBRARY)

    # Construct reconstruction directory path
    recons_dir = f"{SAVE_PATH}/samples_sampling{SAMPLING}_nwalkers{N_WALKERS}_nsteps{N_STEPS}_stepsize{STEP_SIZE:.0f}_bounds{BOUNDS}_intens{INTENS_METHOD}_temp{TEMP:.0f}"
    print(f"Reading reconstructions from: {recons_dir}")
    
    # Storage for results
    results = {
        'event_id': [],
        'true_azim': [],
        'true_zenith': [],
        'energy_nu': [],
        'energy_tau': [],
        'angular_error_true': [],
        'angular_error_layout': [],
        'angle_true_layout': [],
        'eigenvalue_ratio': [],
        'recons_azim': [],
        'recons_zenith': [],
        'layout_azim': [],
        'layout_zenith': [],
        'distance_to_line': []
    }

    # Parameters dict for event processing
    params = {
        'F_MIN': F_MIN,
        'F_MAX': F_MAX,
        'NOISE_STD': NOISE_STD,
        'JITTER_STD': JITTER_STD
    }

    # Process each event
    processed_count = 0
    for event_i in range(n_events):
        if process_event(event_i, tau_events, antennas, efields, sttimes, recons_dir, results, params, plot_flag=args.plot):
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} events")
        else:
            if os.path.exists(os.path.join(recons_dir, f'event{event_i}.npz')):
                print(f"Skipping event {event_i}: reconstruction file found but processing failed")

    # Save and analyze results
    df = save_results_to_csv(results, SAVE_PATH, SAMPLING, N_WALKERS, N_STEPS)
    print(f"\nSuccessfully processed {len(df)} events")

    # Print statistics
    print_statistics(df)

    # Plot comparison histograms if requested (in addition to event-wise plots)
    print("\nGenerating final comparison plots...")
    df_trigged = pd.read_csv("results_adf/LEevents_phased_snr.csv")
    df = df.merge(df_trigged[['event_id', 'snr']], on='event_id', how='left')
    df = df[df['snr'] >= 5]
    plot_results(df, SAVE_PATH, SAMPLING, N_WALKERS, N_STEPS)

    print("\nDone!")


if __name__ == "__main__":
    main()
