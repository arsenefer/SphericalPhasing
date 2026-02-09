import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dataclasses import dataclass
from time import time
from itertools import product
# Importing custom modules
import beamforming.beamforming_module_para as bm
import beamforming.read_sims_module as rm
import beamforming.signal_module as sm
import beamforming.sampling_module_para as sp
import beamforming.parameter_reconstruction as rec

from beamforming.adf_weights import compute_adf_weights

import argparse

R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/data/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf_bruitfilt'
    sampling: str = 'random'
    n_walkers: int = 200
    n_steps: int = 400
    step_size: float = 1000.
    f_min: float = 30
    f_max: float = 250
    noise_std: float = 0
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

def parse_args():
    """
    Parse command line arguments to override default configuration.
    """
    parser = argparse.ArgumentParser(description="Reconstruction Configuration")
    parser.add_argument('--path_to_library', type=str, help='Path to the simulation library', default=None)
    parser.add_argument('--save_path', type=str, help='Path to save results', default=None)
    parser.add_argument('--f_min', type=float, help='Minimum frequency for bandpass filter', default=30)
    parser.add_argument('--f_max', type=float, help='Maximum frequency for bandpass filter', default=250)
    parser.add_argument('--noise_std', type=float, help='Standard deviation of noise to add', default=0)
    parser.add_argument('--jitter_std', type=float, help='Standard deviation of timing jitter to add', default=0)
    parser.add_argument('--intens_method', type=str, choices=['amplitude', 'power'], help='Method for intensity calculation', default='amplitude')
    parser.add_argument('--norm_weights', type=bool, help='Whether to normalize ADF weights', default=False)
    
    args = parser.parse_args()
    config = Config()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    
    return config


def compute_intensity_map(theta, phi, Xsource, xant, yant, zant, signals, config):
    shifted_times = bm.spherical_phasing(Xsource[0], Xsource[1], Xsource[2], xant, yant, zant, signals[0, :, 0])
    range_theta = np.linspace(-4, 4, 100) * np.pi / 180 + theta
    range_phi = np.linspace(-4, 4, 100) * np.pi / 180 + phi
    All_intensity = np.zeros((len(range_theta), len(range_phi)))
    for i, theta_i in enumerate(range_theta):
        for j, phi_j in enumerate(range_phi):
            weights = compute_adf_weights(Xsource[0], Xsource[1], Xsource[2], theta_i, phi_j, xant, yant, zant, delta_omega=1.0, norm_weights=config.norm_weights)
            output = bm.signals_summing_adf(signals, shifted_times, weights[None, :])
            norm_2 = output[0, 1, :]**2 + output[0, 2, :]**2 + output[0, 3, :]**2
            compute_intensity = sp.compute_amp if config.intens_method == 'amplitude' else sp.compute_power
            intensity = compute_intensity(norm_2)
            All_intensity[i, j] = intensity
    return All_intensity, range_theta, range_phi

def main(config: Config):
    if "2e7" in config.path_to_library:
        name_prefix = "LEevents_"
    elif "1e8" in config.path_to_library:
        name_prefix = "HEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = range(0,efields.shape[0])  # List of events to process
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

        Xsource = xmax_pos
        All_intensity, range_theta, range_phi = compute_intensity_map(theta, phi, 
                                                                      Xsource, 
                                                                      xant, yant, zant, 
                                                                      signals, 
                                                                      config)
                                
        amps = np.max( np.linalg.norm(signals[1:], axis=0), axis=-1)
        k_layout = np.average(antenna_pos, axis=0, weights=amps) - xmax_pos
        k_layout /= np.linalg.norm(k_layout)
        theta_layout = np.arccos(-k_layout[2])
        phi_layout = np.arctan2(-k_layout[1], -k_layout[0])
        max_idx = np.unravel_index(np.argmax(All_intensity), All_intensity.shape)
        theta_max = range_theta[max_idx[0]]
        phi_max = range_phi[max_idx[1]]
        k_rec = -np.array([
            np.sin(theta_max) * np.cos(phi_max),
            np.sin(theta_max) * np.sin(phi_max),
            np.cos(theta_max)
        ])
        results.append({
            'event_id': eventi,
            'true_phi': phi,
            'true_theta': theta,
            'layout_phi': phi_layout,
            'layout_theta': theta_layout,
            'max_intensity': np.max(All_intensity),
            'reco_phi': phi_max,
            'reco_theta': theta_max,
            "theta_error_deg": (theta_max - theta) * R2D,
            "phi_error_deg": (phi_max - phi) * R2D,
            "angular_error_deg": np.arccos((shower_dir * k_rec).sum()) * R2D,
            "xmax_pos_x": xmax_pos[0],
            "xmax_pos_y": xmax_pos[1],
            "xmax_pos_z": xmax_pos[2],
            "tau_pos_x": tau_pos[0],
            "tau_pos_y": tau_pos[1],
            "tau_pos_z": tau_pos[2],
            'max_amp': np.max(amps),
            "ave_amp": np.mean(amps),
            "nb_above5sigma": np.sum(amps > 5 * config.noise_std),
            "nb_above3sigma": np.sum(amps > 3 * config.noise_std),
            "nb_above1sigma": np.sum(amps > 1 * config.noise_std),
            "energy": en_tau,
            "en_nu": en_nu,
        })
        np.save(os.path.join(config.save_path, f'{name_prefix}{eventi}_intensity_map.npy'), All_intensity)
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results.csv'), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results.csv'), index=False)
    t_end = time()
    print(f"Reconstruction completed in {t_end - t_0:.2f} seconds for {Nevents} events.")
    plt.hist(df['angular_error_deg'], bins=20, edgecolor='black')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Number of Events')
    plt.title('Angular Error Distribution')
    plt.figure()
    plt.hist(df['theta_error_deg'], bins=20, edgecolor='black', alpha=0.7, label='Theta Error')
    plt.hist(df['phi_error_deg'], bins=20, edgecolor='black', alpha=0.7, label='Phi Error')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Number of Events')
    plt.title('Theta and Phi Error Distribution')
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    config = parse_args()
    main(config)