import sys
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

# Local modules
from . import beamforming_module_para as bm
from .utils import progressbar
from .update_walker import mcmc_candidate, temper_candidate

# Sampling frequency in GHz
F_SAMPLE = 10.0  # GHz

def compute_amp(signal_mod_squared):
    """
    Compute the amplitude of the signal.

    Parameters:
        signal_mod_squared (ndarray): Squared modulus of the signal.

    Returns:
        float: Amplitude of the signal.
    """
    return np.sqrt(np.max(signal_mod_squared))

def compute_power(signal_mod_squared):
    """
    Compute the power of the signal.

    Parameters:
        signal_mod_squared (ndarray): Squared modulus of the signal.

    Returns:
        float: Power of the signal.
    """
    return np.sum(signal_mod_squared) / F_SAMPLE

################################################################################
# Spherical beamforming

def sample_MCMC(xant, yant, zant, signals, bound_function, X_0, n_steps=100, method='amplitude', step_size=1.0, T=1.0, reference_antenna=None):
    """
    Perform Metropolis-Hastings sampling on walker positions using beamforming.

    Parameters:
        xant, yant, zant (ndarray): Antenna positions.
        signals (ndarray): Input signals.
        bound_function (callable): Function to enforce bounds on walker positions.
        X_0 (ndarray): Initial walker positions.
        n_steps (int): Number of sampling steps.
        method (str): Method to compute intensity ('amplitude' or 'power').
        step_size (float): Step size for MCMC.
        T (float): Temperature for acceptance ratio.
        reference_antenna (int): Reference antenna index.

    Returns:
        tuple: All walker positions and their corresponding signal intensities.
    """
    n_walkers = X_0.shape[0]
    if method not in {"amplitude", "power"}:
        raise NotImplementedError('Method should be "amplitude" or "power"')
    compute_intensity = compute_amp if method == 'amplitude' else compute_power

    all_walker_pos = np.zeros((n_steps, n_walkers, 3))
    all_walker_pos[0, :, :] = X_0.copy()

    signal_intensity = np.zeros((n_steps, n_walkers))

    # Initialize
    shifted_times, list_summed_signals = bm.spherical_beamforming(
        *(all_walker_pos[0, :, :].T), xant, yant, zant, signals, reference_antenna=reference_antenna
    )
    signal_intensity[0, :] = np.array([
        compute_intensity(sig[1, :]**2 + sig[2, :]**2 + sig[3, :]**2) for sig in list_summed_signals
    ])

    for i in range(1, n_steps):
        candidate_walkers_pos = mcmc_candidate(
            all_walker_pos[i - 1, :, :], step_size=step_size, bound_function=bound_function
        )

        shifted_times, list_summed_signals = bm.spherical_beamforming(
            *(candidate_walkers_pos.T), xant, yant, zant, signals, reference_antenna=reference_antenna
        )
        candidate_intensities = np.array([
            compute_intensity(sig[1, :]**2 + sig[2, :]**2 + sig[3, :]**2) for sig in list_summed_signals
        ])

        acceptance_ratios = np.exp((candidate_intensities - signal_intensity[i - 1, :]) / T)
        u = np.random.rand(len(acceptance_ratios))
        accepted = u <= acceptance_ratios

        signal_intensity[i, accepted] = candidate_intensities[accepted]
        signal_intensity[i, ~accepted] = signal_intensity[i - 1, ~accepted]

        all_walker_pos[i, accepted, :] = candidate_walkers_pos[accepted]
        all_walker_pos[i, ~accepted, :] = all_walker_pos[i - 1, ~accepted]

    return all_walker_pos, signal_intensity

def sample_temper(xant, yant, zant, signals, bound_function, X_0, n_steps=100, method='amplitude', step_size=1.0, T=1.0, reference_antenna=None):
    """
    Perform sampling using simulated tempering.

    Parameters:
        xant, yant, zant (ndarray): Antenna positions.
        signals (ndarray): Input signals.
        bound_function (callable): Function to enforce bounds on walker positions.
        X_0 (ndarray): Initial walker positions.
        n_steps (int): Number of sampling steps.
        method (str): Method to compute intensity ('amplitude' or 'power').
        step_size (float): Step size for tempering.
        T (float): Temperature for acceptance ratio.
        reference_antenna (int): Reference antenna index.

    Returns:
        tuple: All walker positions and their corresponding signal intensities.
    """
    n_walkers = X_0.shape[0]
    current_walkers_pos = X_0.copy()

    all_walker_pos = np.zeros((n_steps, n_walkers, 3))
    signal_intensity = np.zeros((n_steps, n_walkers))
    if method not in {"amplitude", "power"}:
        raise NotImplementedError('Method should be "amplitude" or "power"')
    compute_intensity = compute_amp if method == 'amplitude' else compute_power

    for i in range(n_steps):
        all_walker_pos[i] = current_walkers_pos

        shifted_times, list_summed_signals = bm.spherical_beamforming(
            *(current_walkers_pos.T), xant, yant, zant, signals, reference_antenna=reference_antenna
        )
        signal_intensity[i, :] = np.array([
            compute_intensity(sig[1, :]**2 + sig[2, :]**2 + sig[3, :]**2) for sig in list_summed_signals
        ])
        current_walkers_pos = temper_candidate(
            current_walkers_pos, signal_intensity[i], bound_function, step_size=step_size, T=T
        )

    return all_walker_pos, signal_intensity

def sample_random_grid(xant, yant, zant, signals, x_range, y_range, z_range):
    """
    Perform random grid sampling for beamforming.

    Parameters:
        xant, yant, zant (ndarray): Antenna positions.
        signals (ndarray): Input signals.
        x_range, y_range, z_range (ndarray): Ranges for x, y, and z coordinates.

    Returns:
        tuple: Best source coordinates, beamformed maximum map, and best index.
    """
    beamformed_maximum_map = np.zeros((len(x_range), len(y_range), len(z_range)))

    for _ in progressbar(range(1000), prefix="", size=60, out=sys.stdout):
        i, j, k = np.random.randint(0, len(z_range), size=3)

        phased_Sig_spherical, beamformed_Sig_spherical = bm.spherical_beamforming(
            x_range[i], y_range[j], z_range[k], xant, yant, zant, signals
        )
        spherical_beamformed_Sig_modulus = np.sqrt(
            beamformed_Sig_spherical[1, :]**2 + beamformed_Sig_spherical[2, :]**2 + beamformed_Sig_spherical[3, :]**2
        )
        beamformed_maximum_map[i, j, k] = np.amax(spherical_beamformed_Sig_modulus)

    best_rec = np.unravel_index(beamformed_maximum_map.argmax(), beamformed_maximum_map.shape)
    best_xsrc, best_ysrc, best_zsrc = x_range[best_rec[0]], y_range[best_rec[1]], z_range[best_rec[2]]

    return best_xsrc, best_ysrc, best_zsrc, beamformed_maximum_map, best_rec

def presampled_beamforming(xant, yant, zant, signals, walker_pos, method='amplitude', T=1.0, reference_antenna=None):
    """
    Perform beamforming on pre-sampled walker positions.

    Parameters:
        xant, yant, zant (ndarray): Antenna positions.
        signals (ndarray): Input signals.
        walker_pos (ndarray): Pre-sampled walker positions.
        method (str): Method to compute intensity ('amplitude' or 'power').
        T (float): Temperature for acceptance ratio.
        reference_antenna (int): Reference antenna index.

    Returns:
        tuple: Walker positions and their corresponding signal intensities.
    """
    signal_intensity = np.zeros((walker_pos.shape[0], walker_pos.shape[1])) * np.nan
    if method not in {"amplitude", "power"}:
        raise NotImplementedError('Method should be "amplitude" or "power"')
    compute_intensity = compute_amp if method == 'amplitude' else compute_power

    for i, current_walkers_pos in enumerate(walker_pos):
        available_mask = ~np.isnan(current_walkers_pos).any(axis=1)
        available_walker_pos = current_walkers_pos[available_mask]

        shifted_times, list_summed_signals = bm.spherical_beamforming(
            *(available_walker_pos.T), xant, yant, zant, signals, reference_antenna=reference_antenna
        )
        signal_intensity[i, available_mask] = np.array([
            compute_intensity(sig[1, :]**2 + sig[2, :]**2 + sig[3, :]**2) for sig in list_summed_signals
        ])

    return walker_pos, signal_intensity