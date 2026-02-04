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
    noise_std: float = 13
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

def main(config: Config):
    name_prefix = "LEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = range(0,250)  # List of events to process
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
        shifted_times = bm.spherical_phasing(Xsource[0], Xsource[1], Xsource[2], xant, yant, zant, signals[0, :, 0])
        range_theta = np.linspace(-4, 4, 100) * np.pi / 180 + theta
        range_phi = np.linspace(-4, 4, 100) * np.pi / 180 + phi
        All_intensity = np.zeros((len(range_theta), len(range_phi)))
        for i, theta_i in enumerate(range_theta):
            for j, phi_j in enumerate(range_phi):
                weights = compute_adf_weights(Xsource[0], Xsource[1], Xsource[2], theta_i, phi_j, xant, yant, zant, delta_omega=1.0)
                output = bm.signals_summing_adf(signals, shifted_times, weights[None, :])
                norm_2 = output[0, 1, :]**2 + output[0, 2, :]**2 + output[0, 3, :]**2
                compute_intensity = sp.compute_amp if config.intens_method == 'amplitude' else sp.compute_power
                intensity = compute_intensity(norm_2)
                All_intensity[i, j] = intensity
        
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
        

        # fig, ax = plt.subplots(figsize=(8, 6))
        # c = ax.pcolormesh(range_phi * 180 / np.pi, range_theta * 180 / np.pi, All_intensity, shading='auto')
        # ax.scatter(phi*180/np.pi, theta*180/np.pi, color='red', marker='x', label='True Direction')
        # ax.scatter(phi_layout*180/np.pi, theta_layout*180/np.pi, color='blue', marker='o', label='Layout Direction')
        # ax.set_xlabel('Phi (degrees)')
        # ax.set_ylabel('Theta (degrees)')
        # ax.set_title(f'Beamformed Intensity Map for Event {eventi}')
        # fig.colorbar(c, ax=ax, label='Intensity')
        # plt.figure(figsize=(8,6))
        # plt.scatter(xant/1e3, yant/1e3, c=10*np.log10(amps), cmap='viridis', s=20)
        # plt.xlabel('X Antenna Position (km)')
        # plt.ylabel('Y Antenna Position (km)')
        # plt.title(f'Antenna Layout with Signal Amplitudes for Event {eventi}')
        # plt.colorbar(label='Max Signal Amplitude')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

if __name__ == "__main__":
    config = Config()
    main(config)