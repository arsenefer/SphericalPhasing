"""
Perform Monte Carlo reconstructions using the ADF beamforming method on a set of simulated events. This script reads in a library of simulated events, applies noise and timing jitter, and then reconstructs the shower direction using the ADF method. The results are saved to a specified directory for further analysis.
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from tqdm import tqdm
sys.path.insert(0, '/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')
sys.path.insert(0, '/pbs/home/a/aferrier/WorkingDir/HERON/SphericalPhasing')
from dataclasses import dataclass
from time import time
from itertools import product
# Importing custom modules
import beamforming_module.beamforming_module_para as bm
import beamforming_module.read_sims_module as rm
import beamforming_module.signal_module as sm
import beamforming_module.sampling_module_para as sp
import beamforming_module.parameter_reconstruction as rec

from beamforming_module.adf_weights import ADF_parameters, prior_adf, shower_direction_vector

import argparse

import emcee
from scipy.optimize import differential_evolution
R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf/test_fixed'
    sampling: str = 'random'
    f_min: float = 50
    f_max: float = 200
    noise_std: float = 13
    jitter_std: float = 0
    delta_omega: float = 1.2
    intens_method: str = 'amplitude'  # Intensity calculation method
    norm_weights: bool = True         # Whether to normalize ADF weights
    prior: bool = True               # Whether to add ADF prior to intensity
    constant: bool = True             # Whether to add constant term to ADF weights


def parse_args():
    """
    Parse command line arguments to override default configuration.
    """
    parser = argparse.ArgumentParser(description="Reconstruction Configuration")
    parser.add_argument('--path_to_library', type=str, help='Path to the simulation library', default=None)
    parser.add_argument('--save_path', type=str, help='Path to save results', default=None)
    parser.add_argument('--f_min', type=float, help='Minimum frequency for bandpass filter', default=None)
    parser.add_argument('--f_max', type=float, help='Maximum frequency for bandpass filter', default=None)
    parser.add_argument('--noise_std', type=float, help='Standard deviation of noise to add', default=None)
    parser.add_argument('--jitter_std', type=float, help='Standard deviation of timing jitter to add', default=None)
    parser.add_argument('--intens_method', type=str, choices=['amplitude', 'power'], help='Method for intensity calculation', default=None)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--norm_weights', type=str2bool, help='Whether to normalize ADF weights', default=None)
    parser.add_argument('--prior', type=str2bool, help='Whether to add ADF prior to intensity', default=None)
    parser.add_argument('--constant', type=str2bool, help='Whether to add constant term to ADF weights', default=None)
    
    args = parser.parse_args()
    config = Config()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
            print(key, value)
    return config

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
def montain_prior(theta):
    prior_montain = _sigmoid(2 * ((theta - np.pi/2) * R2D + 0.5))
    return prior_montain*100

def log_p(v, xant, yant, zant, signals, config):
    """
    Compute the logarithm of the probability, handling zero values to avoid log(0).
    """
    Xsource, theta, phi = v[:3], v[3], v[4]
    shifted_times = bm.spherical_phasing(Xsource[0], Xsource[1], Xsource[2], xant, yant, zant, signals[0, :, 0])
    eta, omega, omega_cr, l_ant, weights = ADF_parameters(
                theta, phi, 
                delta_omega=config.delta_omega, 
                Xants = np.stack((xant, yant, zant), axis=-1), 
                Xsource=Xsource,
                norm_weights=config.norm_weights, constant=config.constant)
    
    output = bm.signals_summing_adf(signals, shifted_times, weights[None, :])
    norm_2 = output[0, 1, :]**2 + output[0, 2, :]**2 + output[0, 3, :]**2
    compute_intensity = sp.compute_amp if config.intens_method == 'amplitude' else sp.compute_power
    intensity = compute_intensity(norm_2)
    if config.prior:
        intensity += prior_adf(weights, omega, omega_cr)
        intensity += montain_prior(theta)
    return intensity/config.noise_std
def distance_to_line(x, x0, v):

    """Calculate distance from point x to line defined by point x0 and direction v."""
    v_norm = v / np.linalg.norm(v)
    return np.linalg.norm(np.cross(x - x0, v_norm))
def make_result_dict(eventi, best_fit_params, best_fit_intensity, xmax_pos, theta, phi, verbose=True):
    recons_xmax, recons_theta, recons_phi = best_fit_params[:3], best_fit_params[3], best_fit_params[4]
    if verbose:
        print(f"Event {eventi}:")
        print(f"True parameters: Xmax: {xmax_pos}, Theta: {theta*R2D}, Phi: {phi*R2D}")
        print(f"Reconstructed parameters: Xmax: {recons_xmax}, Theta: {recons_theta*R2D}, Phi: {recons_phi*R2D}")
        print(f"Parameter errors: Xmax error: {np.linalg.norm(recons_xmax - xmax_pos)}, Theta error: {(recons_theta - theta)*R2D}, Phi error: {(recons_phi - phi)*R2D}")

    k_true = shower_direction_vector(theta, phi)
    k_recons = shower_direction_vector(recons_theta, recons_phi)
    angle_error = np.arccos(np.clip(np.dot(k_true, k_recons), -1.0, 1.0)) * R2D
    theta_error = (recons_theta - theta)*R2D
    phi_error = (recons_phi - phi)*R2D
    xmax_error = np.linalg.norm(recons_xmax - xmax_pos)

    xmax_lat_error = distance_to_line(recons_xmax, xmax_pos, k_true)
    xmax_long_error = np.dot(recons_xmax - xmax_pos, k_true)

    return {
        'event_index': eventi,
        'best_fit_xmax': recons_xmax,
        'best_fit_theta': recons_theta,
        'best_fit_phi': recons_phi,
        'best_fit_intensity': best_fit_intensity,
        'true_xmax': xmax_pos,
        'true_theta': theta,
        'true_phi': phi,
        'xmax_error': xmax_error,
        'theta_error': theta_error, 
        'phi_error': phi_error,
        'angle_error': angle_error,
        'xmax_lat_error': xmax_lat_error,
        'xmax_long_error': xmax_long_error
    }


def process_event(eventi):
    config = globals().get('CONFIG')
    TAU_EVENTS = globals().get('TAU_EVENTS')
    ANTENNAS = globals().get('ANTENNAS')
    EFIELDS = globals().get('EFIELDS')
    STTIMES = globals().get('STTIMES')

    np.random.seed(eventi)

    # Read event data; use actual event index 'eventi'
    (_azim, _zen, en_nu, en_tau, tau_pos, xmax_pos, _, antenna_pos,
        xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
        eventi, TAU_EVENTS, ANTENNAS, EFIELDS, STTIMES, config.f_min, config.f_max, config.noise_std, config.jitter_std
    )

    # Initial guess for optimization (can be improved with a more informed guess)
    Xsource_guess = xmax_pos
    theta_guess = theta
    phi_guess = phi
    initial_guess = np.array([Xsource_guess[0], Xsource_guess[1], Xsource_guess[2], theta_guess, phi_guess])
    # Run optimization to find best-fit parameters with differential evolution
    bounds = [(Xsource_guess[0] - 20000, Xsource_guess[0] + 20000), 
              (Xsource_guess[1] - 20000, Xsource_guess[1] + 20000), 
              (Xsource_guess[2] - 5000, Xsource_guess[2] + 5000), 
              (theta_guess-4/R2D, theta_guess+4/R2D), 
              (phi_guess -4/R2D, phi_guess+4/R2D)]
    result = differential_evolution(lambda v: -log_p(v, xant, yant, zant, signals, config), 
                                    bounds,
                                    popsize=15, maxiter=2000, tol=1e-6)
    best_fit_params = result.x
    best_fit_intensity = -result.fun
    verbose = 'volatile' in os.getcwd()
    result = make_result_dict(eventi, best_fit_params, best_fit_intensity, xmax_pos, theta, phi, verbose=verbose)
    return result

def main(config: Config):
    print(f"Using configuration: {config}")
    if "2e7" in config.path_to_library:
        name_prefix = "LEevents_"
        n_events = 972
    elif "1e8" in config.path_to_library:
        name_prefix = "HEevents_"
        n_events = 250
    # Load library once (large arrays will be inherited by forked workers)
    global TAU_EVENTS, ANTENNAS, EFIELDS, STTIMES, CONFIG
    TAU_EVENTS, ANTENNAS, EFIELDS, STTIMES = rm.read_library(config.path_to_library)
    CONFIG = config
    Eventlist = range(0, len(TAU_EVENTS))  # List of events to process
    results = []
    if "volatile" in os.getcwd():
        # Running on local machine, process events sequentially
        for i, eventi in tqdm(enumerate(Eventlist), total=len(Eventlist)):
            result = process_event(eventi)
            # Save or log the result as needed (e.g., append to a list, save to file, etc.)
            results.append(result)
            if (i-1) % 20 == 0:
                results_df = pd.DataFrame(results)
                os.makedirs(config.save_path, exist_ok=True)
                results_df.to_csv(os.path.join(config.save_path, f"{name_prefix}reconstruction_results.csv"), index=False)
    else:
        # Running on cluster, use multiprocessing to speed up processing
        import multiprocessing
        with multiprocessing.Pool(processes=32) as pool:  
            for i, result in enumerate(tqdm(pool.imap_unordered(process_event, Eventlist), total=len(Eventlist))):
                results.append(result)
                if (i-1) % 20 == 0:
                    results_df = pd.DataFrame(results)
                    os.makedirs(config.save_path, exist_ok=True)
                    results_df.to_csv(os.path.join(config.save_path, f"{name_prefix}reconstruction_results.csv"), index=False)

    # Save results to a file or process further as needed
    results_df = pd.DataFrame(results)
    os.makedirs(config.save_path, exist_ok=True)
    results_df.to_csv(os.path.join(config.save_path, f"{name_prefix}reconstruction_results.csv"), index=False)
    
if __name__ == "__main__":
    config = parse_args()
    main(config)