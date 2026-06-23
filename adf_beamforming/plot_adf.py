import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dataclasses import dataclass
from time import time
from itertools import product
# Importing custom modules
import beamforming_module.beamforming_module_para as bm
import beamforming_module.read_sims_module as rm
import beamforming_module.signal_module as sm
import beamforming_module.sampling_module_para as sp
import beamforming_module.parameter_reconstruction as rec

 

R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf'
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

def main(config: Config):
    name_prefix = "LEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = range(0, 250)  # List of events to process
    Nevents = len(Eventlist)

    results = []

    df = pd.read_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results.csv'))
    mean_amplitudes = []
    max_amplitudes = []
    mean_power = []
    max_power = []

    plt.hist(df['angular_error_deg'], bins=50, edgecolor='black')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Number of Events')
    plt.title(f'Angular Error Distribution : $\\mu$={df["angular_error_deg"].mean():.2f}°')
    plt.figure()
    plt.hist(df['theta_error_deg'], bins=50, edgecolor='black', alpha=0.7, label=f'$\\theta$: $\\mu$={df["theta_error_deg"].mean():.2f}, $\\sigma$={df["theta_error_deg"].std():.2f}')
    plt.hist(df['phi_error_deg'], bins=50, edgecolor='black', alpha=0.7, label=f'$\\phi$: $\\mu$={df["phi_error_deg"].mean():.2f}, $\\sigma$={df["phi_error_deg"].std():.2f}')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Number of Events')
    plt.title('Theta and Phi Error Distribution')
    plt.legend()
    plt.show()
    
    plt.scatter(df['mean_amplitude'], df['angular_error_deg'], alpha=0.7)
    plt.xlabel('Mean Amplitude (a.u.)')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Error vs Mean Amplitude')
    plt.figure()
    plt.scatter(df['max_amplitude'], df['angular_error_deg'], alpha=0.7)
    plt.xlabel('Max Amplitude (a.u.)')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Error vs Max Amplitude')
    plt.show()
    
    plt.scatter(df['mean_power'], df['angular_error_deg'], alpha=0.7)
    plt.xlabel('Mean power (a.u.)')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Error vs Mean power')
    plt.figure()
    plt.scatter(df['max_power'], df['angular_error_deg'], alpha=0.7)
    plt.xlabel('Max power (a.u.)')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Angular Error vs Max power')
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