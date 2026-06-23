import numpy as np
from scipy.special import logsumexp
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

from beamforming_module.adf_weights import ADF_parameters, prior_adf
from beamforming_module.utils import cart2sph, sph2cart
from adf_beamforming.explo_adf import compute_intensity_map
import argparse
from tqdm import tqdm
R2D = 180. / np.pi  # Conversion factor from radians to degrees
import multiprocessing

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf/full_recons/'
    interf_path: str = 'results/samples_samplingrandom_nwalkers200_nsteps600_stepsize1000_boundsflat_sphere_intensamplitude_temp5000'
    sampling: str = 'random'
    f_min: float = 50
    f_max: float = 200
    noise_std: float = 13
    jitter_std: float = 0
    delta_omega: float = 1.2
    intens_method: str = 'amplitude'  # Intensity calculation method
    norm_weights: bool = True         # Whether to normalize ADF weights
    prior: bool = False               # Whether to add ADF prior to intensity
    constant: bool = True             # Whether to add constant term to ADF weights


def parse_args():
    """
    Parse command line arguments to override default configuration.
    """
    parser = argparse.ArgumentParser(description="Reconstruction Configuration")
    parser.add_argument('--path_to_library', type=str, help='Path to the simulation library', default=None)
    parser.add_argument('--save_path', type=str, help='Path to save results', default=None)
    parser.add_argument('--interf_path', type=str, help='Path to save results', default=None)
    parser.add_argument('--f_min', type=float, help='Minimum frequency for bandpass filter', default=50)
    parser.add_argument('--f_max', type=float, help='Maximum frequency for bandpass filter', default=200)
    parser.add_argument('--noise_std', type=float, help='Standard deviation of noise to add', default=13)
    parser.add_argument('--jitter_std', type=float, help='Standard deviation of timing jitter to add', default=0)
    parser.add_argument('--intens_method', type=str, choices=['amplitude', 'power'], help='Method for intensity calculation', default='amplitude')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--norm_weights', type=str2bool, help='Whether to normalize ADF weights', default=True)
    parser.add_argument('--prior', type=str2bool, help='Whether to add ADF prior to intensity', default=False)
    parser.add_argument('--constant', type=str2bool, help='Whether to add constant term to ADF weights', default=True)
    
    args = parser.parse_args()
    config = Config()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
            print(key, value)
    return config

def compute_candidate_source_points(walker_positions, walker_intensities, n_candidates=10, quantile=0.98):
    """
    Compute candidate source points from interferometric samples.
    
    Inputs:
        walker_positions: (N, 3) array of candidate source positions
        walker_intensities: (N,) array of corresponding intensities
        n_candidates: number of top candidates to return
    
    Returns:
        candidate_source_points: (n_candidates, 3) array of top candidate source positions
    """
    intensity_threshold = np.quantile(walker_intensities.flatten(), quantile)
    good_samples_mask = walker_intensities.flatten() >= intensity_threshold
    good_positions = walker_positions.reshape(-1, 3)[good_samples_mask]
    good_intensities = walker_intensities.flatten()[good_samples_mask]
    w = np.asarray(good_intensities, dtype=float)
    w = w / w.sum()
    
    center = np.average(good_positions, axis=0, weights=w)
    Xc = good_positions - center
    
    cov = (Xc * w[:, None]).T @ Xc
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    max_eigval = np.sqrt(np.max(eigvals))
    interf_direction = eigvecs[:, np.argmax(eigvals)]
    interf_direction = interf_direction / np.linalg.norm(interf_direction)
    approx_layout_dir = -center/np.linalg.norm(center)
    interf_direction = interf_direction if np.dot(interf_direction, approx_layout_dir) > 0 else -interf_direction

    t_values = np.linspace(-max_eigval, max_eigval, n_candidates)
    center_point = np.average(good_positions, axis=0, weights=w**8)
    points_along_direction = center_point[np.newaxis, :] + interf_direction[np.newaxis, :] * t_values[:, np.newaxis]
    return points_along_direction

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
def montain_prior(theta_range, phi_range):
    thetas = theta_range[:, None] * np.ones((1, len(phi_range)))
    prior_montain = _sigmoid(2 * ((thetas - np.pi/2) * R2D + 0.5))
    return prior_montain*100


def get_name_prefix(path_to_library: str):
    if "2e7" in path_to_library:
        return "LEevents_"
    if "1e8" in path_to_library:
        return "HEevents_"
    return "events_"


def process_single_event(eventi):
    """Top-level worker to process a single event. Uses module-level CONFIG and TAU_* globals."""
    config = globals().get('CONFIG')
    TAU_EVENTS = globals().get('TAU_EVENTS')
    ANTENNAS = globals().get('ANTENNAS')
    EFIELDS = globals().get('EFIELDS')
    STTIMES = globals().get('STTIMES')
    if config is None:
        raise RuntimeError('CONFIG not set in module globals')
    try:
        np.random.seed(eventi)
        interferometric_samples = np.load(f"{config.interf_path}/event{eventi}.npz")
        walker_positions = interferometric_samples['walker_positions']
        # support two possible intensity keys
        walker_intensities = interferometric_samples.get('walker_intensities') or interferometric_samples.get('beamformed_intensity_array')
        candidate_source_points = compute_candidate_source_points(walker_positions, walker_intensities, n_candidates=10)
        del interferometric_samples, walker_positions, walker_intensities

        # Read event data; use actual event index 'eventi'
        (_azim, _zen, en_nu, en_tau, tau_pos, xmax_pos, _, antenna_pos,
         xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
            eventi, TAU_EVENTS, ANTENNAS, EFIELDS, STTIMES, config.f_min, config.f_max, config.noise_std, config.jitter_std
        )

        maps = []
        for Xsource in candidate_source_points:
            All_intensity, range_theta, range_phi = compute_intensity_map(theta, phi,
                                                                        Xsource,
                                                                        xant, yant, zant,
                                                                        signals,
                                                                        config)
            maps.append(All_intensity)
        maps = np.array(maps)
        prior_m = montain_prior(range_theta, range_phi)

        def treat_maps_local(maps_local):
            amps = np.max(np.linalg.norm(signals[1:], axis=0), axis=-1)
            k_layout = np.average(antenna_pos, axis=0, weights=amps) - xmax_pos
            k_layout /= np.linalg.norm(k_layout)
            theta_layout, phi_layout = cart2sph(*(-k_layout))

            max_idx = np.unravel_index(np.argmax(maps_local), maps_local.shape)
            source_point_max = candidate_source_points[max_idx[0]]
            theta_max = range_theta[max_idx[1]]
            phi_max = range_phi[max_idx[2]]
            max_amp = maps_local[max_idx[0], max_idx[1], max_idx[2]]

            sum_max_idx = np.unravel_index(np.argmax(maps_local.sum(axis=0)), maps_local.shape[1:])
            theta_sum_max = range_theta[sum_max_idx[0]]
            phi_sum_max = range_phi[sum_max_idx[1]]

            logsumexp_max_idx = np.unravel_index(np.argmax(logsumexp(maps_local, axis=0)), maps_local.shape[1:])
            theta_logsumexp_max = range_theta[logsumexp_max_idx[0]]
            phi_logsumexp_max = range_phi[logsumexp_max_idx[1]]
            k_rec = -np.array([
                np.sin(theta_max) * np.cos(phi_max),
                np.sin(theta_max) * np.sin(phi_max),
                np.cos(theta_max)
            ])

            amps_local = np.max(np.linalg.norm(signals[1:], axis=0), axis=-1)
            return {
                'event_id': eventi,
                'true_phi': phi,
                'true_theta': theta,
                'layout_phi': phi_layout,
                'layout_theta': theta_layout,
                'max_intensity': max_amp,
                'reco_phi': phi_max,
                'reco_theta': theta_max,
                'reco_phi_sum': phi_sum_max,
                'reco_theta_sum': theta_sum_max,
                'reco_phi_logsumexp': phi_logsumexp_max,
                'reco_theta_logsumexp': theta_logsumexp_max,
                "theta_error_deg": (theta_max - theta) * R2D,
                "phi_error_deg": (phi_max - phi) * R2D,
                "angular_error_deg": np.arccos((shower_dir * k_rec).sum()) * R2D,
                "xmax_cheaprec_x": source_point_max[0],
                "xmax_cheaprec_y": source_point_max[1],
                "xmax_cheaprec_z": source_point_max[2],
                "xmax_pos_x": xmax_pos[0],
                "xmax_pos_y": xmax_pos[1],
                "xmax_pos_z": xmax_pos[2],
                "tau_pos_x": tau_pos[0],
                "tau_pos_y": tau_pos[1],
                "tau_pos_z": tau_pos[2],
                'max_amp': np.max(amps_local),
                "ave_amp": np.mean(amps_local),
                "nb_above5sigma": np.sum(amps_local > 5 * config.noise_std),
                "nb_above3sigma": np.sum(amps_local > 3 * config.noise_std),
                "nb_above1sigma": np.sum(amps_local > 1 * config.noise_std),
                "energy": en_tau,
                "en_nu": en_nu,
            }

        res = treat_maps_local(maps)
        np.save(os.path.join(config.save_path, f'{get_name_prefix(config.path_to_library)}{eventi}_intensity_map.npy'), maps)

        maps_with_prior = maps + prior_m[None, :, :]
        res_m = treat_maps_local(maps_with_prior)
        np.save(os.path.join(config.save_path, f'{get_name_prefix(config.path_to_library)}{eventi}_intensity_map_montain.npy'), maps_with_prior)

        return res, res_m

    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error processing event {eventi}: {e}")
        return None

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

    Nevents = len(Eventlist)

    results = []
    results_montain_prior = []
    t_0 = time()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # No nested worker here; use top-level `process_single_event` to avoid pickling issues

    # Run pool with fork context so large arrays are inherited without being pickled
    n_procs = 32
    ctx = multiprocessing.get_context('fork')
    print(f"Starting pool with {n_procs} workers")
    with ctx.Pool(processes=n_procs) as pool:
        for i, out in enumerate(tqdm(pool.imap_unordered(process_single_event, Eventlist), total=Nevents)):
            if out is None:
                continue
            res, res_m = out
            results.append(res)
            results_montain_prior.append(res_m)
            if (len(results)) % 50 == 0:
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results.csv'), index=False)
                df = pd.DataFrame(results_montain_prior)
                df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results_montain.csv'), index=False)

    # final save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results.csv'), index=False)
    df = pd.DataFrame(results_montain_prior)
    df.to_csv(os.path.join(config.save_path, f'{name_prefix}reconstruction_results_montain.csv'), index=False)
    t_end = time()
    print(f"Reconstruction completed in {t_end - t_0:.2f} seconds for {Nevents} events.")
    
    # plt.figure()
    # plt.hist(df['angular_error_deg'], bins=20, edgecolor='black')
    # plt.xlabel('Angular Error (degrees)')
    # plt.ylabel('Number of Events')
    # plt.title('Angular Error Distribution')

    # plt.figure()
    # plt.hist(df['theta_error_deg'], bins=20, edgecolor='black', alpha=0.7, label='Theta Error')
    # plt.hist(df['phi_error_deg'], bins=20, edgecolor='black', alpha=0.7, label='Phi Error')
    # plt.xlabel('Angle Error (degrees)')
    # plt.ylabel('Number of Events')
    # plt.title('Theta and Phi Error Distribution')
    # plt.legend()
    # plt.show()
        

if __name__ == "__main__":
    config = parse_args()
    main(config)