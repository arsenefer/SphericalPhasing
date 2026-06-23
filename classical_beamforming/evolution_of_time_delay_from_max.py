import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dataclasses import dataclass
from time import time

# Importing custom modules
sys.path.append('/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')

import beamforming_module.beamforming_module_para as bm
import beamforming_module.read_sims_module as rm
import beamforming_module.signal_module as sm
import beamforming_module.sampling_module_para as sp
import beamforming_module.parameter_reconstruction as rec

from beamforming_module.sampler_init import (
    cubic_bound_function, 
    cubic_init, 
    sphere_bound_function, 
    sphere_init, 
    xyz_parameter_bounds, 
    flat_sphere_bound_function, 
    flat_sphere_init,
    create_regular_grid
)

R2D = 180. / np.pi  # Conversion factor from radians to degrees

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results'
    sampling: str = 'random'
    n_walkers: int = 200
    n_steps: int = 400
    step_size: float = 1000.
    f_min: float = 0
    f_max: float = 5000
    noise_std: float = 0
    jitter_std: float = 0
    bounds: str = 'flat_sphere'  # Options: 'cubic', 'sphere', 'flat_sphere'
    r: float = 10.e3  # Radius for spherical bounds
    thickness: float = 4e3  # Thickness for flat spherical bounds
    intens_method: str = 'amplitude'  # Intensity calculation method
    temp: float = 5000.  # Temperature for sampling
    x_max_method: str = 'max'  # Method for finding maximum
    n_best_walkers: int = 0  # Number of best walkers to consider
    sep_walkers: bool = False  # Separate walkers flag
    burn_in: int = 0  # Number of burn-in steps to discard



def plot_event(signals, antenna_pos, xmax_pos, tau_pos, sampling_points=None):
    """
    Plot the signals from antennas.

    Parameters:
        signals (ndarray): Signal data.
        antenna_pos (ndarray): Antenna positions.
    """
    sampling_period = signals[0, 0, 1] - signals[0, 0, 0]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    amps = np.max(np.linalg.norm(signals[1:, :, :], axis=0), axis=-1)
    axs[0].scatter(antenna_pos[:, 0], antenna_pos[:, 1], c=amps, s=20)
    axs[0].scatter(xmax_pos[0], xmax_pos[1], marker='*', c='r', s=100, label='Xmax Position')
    axs[0].scatter(tau_pos[0], tau_pos[1], marker='*', c='g', s=70, label='Tau Position')
    if sampling_points is not None:
        axs[0].scatter(sampling_points[:, 0], sampling_points[:, 1], c='C1', s=5, zorder=-1)
        axs[0].plot(sampling_points[[0,-1], 0], sampling_points[[0,-1], 1], c='k', lw=.5, zorder=-1)
    axs[0].set_aspect('equal', adjustable='datalim')
    axs[0].set_xlabel('X position (m)')
    axs[0].set_ylabel('Y position (m)')    


    axs[1].scatter(antenna_pos[:, 1], antenna_pos[:, 2], c=amps, s=20)
    axs[1].scatter(xmax_pos[1], xmax_pos[2], marker='*', c='r', s=100, label='Xmax Position')
    axs[1].scatter(tau_pos[1], tau_pos[2], marker='*', c='g', s=70)
    if sampling_points is not None:
        axs[1].scatter(sampling_points[:, 1], sampling_points[:, 2], c='C1', s=5, zorder=-1)
        axs[1].plot(sampling_points[[0,-1], 1], sampling_points[[0,-1], 2], c='k', lw=.5, zorder=-1)
    axs[1].set_aspect('equal', adjustable='datalim')
    axs[1].set_xlabel('Y position (m)')
    axs[1].set_ylabel('Z position (m)') 
    axs[0].legend()
    fig.suptitle('Antenna Layout with Signal Amplitudes and Key Positions')
    return fig, axs

def plot_time_delay(xant, yant, zant, signals, samplepoints, layout_dir, ref_ant=None, fig=None, ax=None):
    """
    Plot time delays for antennas.

    Parameters:
        xant, yant, zant (ndarray): Antenna positions.
        signals (ndarray): Signal data.
        samplepoints (ndarray): Sampling points.
        layout_dir (ndarray): Layout direction vector.
        ref_ant (int): Reference antenna index.
    """
    time_delays = bm.spherical_phasing(
        samplepoints[:,0], samplepoints[:,1], samplepoints[:,2], xant, yant, zant, np.zeros_like(signals[0,:,0])
    )
    amps = np.max(np.linalg.norm(signals[1:, :, :], axis=0), axis=-1)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    
    mid_idx = len(samplepoints) // 2
    if ref_ant is None:
        relative_time_delays = time_delays[:,:] - np.average(time_delays, axis=1, weights=amps, keepdims=True)
        relative_time_delays= relative_time_delays - relative_time_delays[mid_idx]
    else:
        relative_time_delays = time_delays[:,:] - time_delays[:,ref_ant][:,None]
        relative_time_delays= relative_time_delays - relative_time_delays[mid_idx]
    alpha = np.repeat(amps[None,:] / np.max(amps), len(samplepoints), axis=0)
    ax.scatter(
        np.repeat(((samplepoints - np.mean(samplepoints, axis=0)) @ layout_dir)[:,None], len(xant), axis=1),
        relative_time_delays, marker='s', s=10, c=np.repeat(amps[None,:], len(samplepoints), axis=0), cmap='viridis', alpha=alpha**2)
    # for ant_idx in range(len(xant)):
    ax.set_xlabel('Distance from max intens pos along layout direction (m)')
    ax.set_ylabel('Time delay (ns)')
    return fig, ax
    
def plot_amps(samplepoints, list_summed_signals, method='amplitude',fig=None, ax=None, label=None):
    """
    Plot amplitudes of summed signals.

    Parameters:
        samplepoints (ndarray): Sampling points.
        list_summed_signals (list): List of summed signals.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    compute_intensity = sp.compute_amp if method == 'amplitude' else sp.compute_power
    amps = np.array([
            compute_intensity(sig[1, :]**2 + sig[2, :]**2 + sig[3, :]**2) for sig in list_summed_signals
        ])
    ax.plot(
        ((samplepoints - np.mean(samplepoints, axis=0)) @ (samplepoints[-1]-samplepoints[0]) / np.linalg.norm(samplepoints[-1]-samplepoints[0])),
        amps, marker='o', label=label)
    ax.set_xlabel('Distance from max intens pos along chosen direction (m)')
    ax.set_ylabel('Phased intensity (a.u.)')
    return fig, ax

def find_max(config, xant, yant, zant, signals, xmax_pos, method='amplitude'):
    walker_positions = np.array([flat_sphere_init(config.n_walkers, xmax_pos, config.r, config.thickness) for _ in range(config.n_steps)])
    walker_positions, beamformed_intensity_array = sp.presampled_beamforming(
        xant, yant, zant, signals.copy(), walker_positions, 
        method=config.intens_method, T=config.temp, reference_antenna=None
    )
    best_rec = np.unravel_index(beamformed_intensity_array.argmax(), beamformed_intensity_array.shape)
    best_xsrc, best_ysrc, best_zsrc = walker_positions[best_rec[0], best_rec[1]]
    return np.array([best_xsrc, best_ysrc, best_zsrc])


def main(config: Config):
    """
    Main function to perform the reconstruction process.

    Parameters:
        config (Config): Configuration object containing all parameters.
    """
    name_prefix = "LEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = range(5, 20)  # List of events to process
    Nevents = len(Eventlist)

    results = []
    t_0 = time()

    for i, eventi in enumerate(Eventlist):
        np.random.seed(i)  # Set random seed for reproducibility

        # Read event data
        (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos, 
         xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
            eventi, tau_events, antennas, efields, sttimes, config.f_min, config.f_max, config.noise_std, config.jitter_std
        )
        coord_of_max = find_max(config, xant, yant, zant, signals, xmax_pos)
        amps = np.max(np.linalg.norm(signals[1:, :, :], axis=0), axis=-1)
        ref_ant = np.argmax(amps[:])

        layout_dir = np.average(antenna_pos, weights=amps**2, axis=0) - coord_of_max
        layout_dir = layout_dir / np.linalg.norm(layout_dir)
        samplepoints_along_layout = np.linspace(coord_of_max + layout_dir * 3000, coord_of_max - layout_dir * 3000, 100+1, endpoint=True)

        samplepoints_along_shower = np.linspace(coord_of_max + shower_dir * 3000, coord_of_max - shower_dir * 3000, 100+1, endpoint=True)
        fig = plt.figure(figsize=(12, 6))
        # increase vertical spacing between the two rows so xlabels are clearer
        gs = fig.add_gridspec(2, 2, hspace=0.45)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
        ax10 = fig.add_subplot(gs[1, :], sharex=ax00)  # spans both columns on the second row
        axs = np.array([[ax00, ax01], [ax10, ax10]], dtype=object)

        plot_event(signals[:,:], antenna_pos[:], xmax_pos, tau_pos, sampling_points=samplepoints_along_layout)
        plot_time_delay(xant[:], yant[:], zant[:], signals[:,:], samplepoints_along_layout, layout_dir, ref_ant=ref_ant, fig=fig, ax=axs[0,0])
        plot_time_delay(xant[:], yant[:], zant[:], signals[:,:], samplepoints_along_shower, shower_dir, ref_ant=ref_ant, fig=fig, ax=axs[0,1])
        
        shifted_times, list_summed_signals_layout = bm.spherical_beamforming(
            *(samplepoints_along_layout.T), xant, yant, zant, signals
        )
        shifted_times, list_summed_signals_shower = bm.spherical_beamforming(
            *(samplepoints_along_shower.T), xant, yant, zant, signals
        )
        plot_amps(samplepoints_along_layout, list_summed_signals_layout, method=config.intens_method, ax = axs[1,0], label='Along layout direction')
        plot_amps(samplepoints_along_shower, list_summed_signals_shower, method=config.intens_method, ax = axs[1,1], label='Along shower direction')
        
        axs[0,1].set_xlabel('Distance from max intens pos along shower axis (m)')
        axs[0,0].set_title(f'Event {eventi} Layout Direction')
        axs[0,1].set_title(f'Event {eventi} Shower Direction')
        axs[1,0].legend()
        fig.suptitle(f'Event {eventi} Analysis, angle between directions: {np.arccos(np.clip(np.dot(layout_dir, shower_dir), -1.0, 1.0)) * R2D:.2f} deg')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        print("Starting the reconstruction process...")
        config = Config() 
        main(config)
    except KeyboardInterrupt:
        print('Exiting')