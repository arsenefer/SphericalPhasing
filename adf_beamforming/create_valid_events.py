import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import sys, os
from dataclasses import dataclass
from time import time
from itertools import product
# Importing custom modules
sys.path.append('/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')
import beamforming_module.beamforming_module_para as bm
import beamforming_module.read_sims_module as rm
import beamforming_module.signal_module as sm
import beamforming_module.sampling_module_para as sp
import beamforming_module.parameter_reconstruction as rec

 

import argparse

R2D = 180. / np.pi  # Conversion factor from radians to degrees

phased_antennas = [4, 9]

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"
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
    parser.add_argument('--noise_std', type=float, help='Standard deviation of noise to add', default=13)
    parser.add_argument('--jitter_std', type=float, help='Standard deviation of timing jitter to add', default=0)
    parser.add_argument('--intens_method', type=str, choices=['amplitude', 'power'], help='Method for intensity calculation', default='amplitude')
    parser.add_argument('--norm_weights', type=bool, help='Whether to normalize ADF weights', default=False)
    args = parser.parse_args()
    config = Config()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config

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

    snr_on_phased = []
    for i, eventi in enumerate(Eventlist):
        np.random.seed(i)  # Set random seed for reproducibility

        # Read event data
        (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos, 
         xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
                eventi, tau_events, antennas, efields, sttimes, config.f_min, config.f_max, 0, config.jitter_std
            )
        # for ant in range(len(xant)):
        #     plt.text(xant[ant], yant[ant], f'{ant}', fontsize=8, ha='center', va='center')  # Annotate antenna numbers
        #     plt.scatter(xant[ant], yant[ant], s=1)  # Annotate antenna numbers
        # plt.show()
        # exit()
        # print(signals.shape)
        efield_phased = signals[1:, phased_antennas, :]
        hilberd_phased = np.abs(hilbert(efield_phased, axis=-1))
        amp_on_phased_antennas = np.max(np.linalg.norm(hilberd_phased, axis=0), 
                                        axis=-1)  
        # print(config.noise_std, config.f_min, config.f_max)
        snr = np.sqrt(24) * amp_on_phased_antennas / (config.noise_std*np.sqrt(3))
        print(snr)
        # if not trigged:
        #     plt.figure(figsize=(10,6))
        #     plt.plot(hilberd_phased[0, phased_antennas[0], :].T, label='Hilbert Envelope (X Pol)')
        #     plt.plot(hilberd_phased[1, phased_antennas[0], :].T, label='Hilbert Envelope (Y Pol)')
        #     plt.plot(hilberd_phased[2, phased_antennas[0], :].T, label='Hilbert Envelope (Z Pol)')
        #     plt.axhline(5*config.noise_std*np.sqrt(3)/np.sqrt(24), color='r', linestyle='--', label='Trigger Threshold')
        #     plt.title(f'Event {eventi} - No Trigger')
        #     plt.xlabel('Time (ns)')
        #     plt.ylabel('Hilbert Envelope (µV/m)')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()
        snr_on_phased.append(snr)
    snr_on_phased = np.array(snr_on_phased)
    df_phased_snr = pd.DataFrame({'event_id': Eventlist, 'snr': np.max(snr_on_phased, axis=1)})
    for ant in phased_antennas:
        df_phased_snr[f'snr_ant_{ant}'] = snr_on_phased[:, phased_antennas.index(ant)]
    df_phased_snr.to_csv(os.path.join("/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/results_adf", 
                                   f'{name_prefix}phased_snr.csv'), index=False)
    print(f"Processing completed in {time() - t_0:.2f} seconds. Trigged events saved to {config.save_path}.")

if __name__ == "__main__":
    config = parse_args()
    main(config)                          
