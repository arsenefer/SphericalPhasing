import sys
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
#local modules
from . import beamforming_module as bm
from ..utils import progressbar
from ..update_walker import mcmc_candidate, temper_candidate
F_SAMPLE = 10. #GHz

def _compute_amp(spherical_beamformed_Sig_modulus_2):
    return np.sqrt(np.max(spherical_beamformed_Sig_modulus_2))
def _compute_power(spherical_beamformed_Sig_modulus_2):
            return np.sum(spherical_beamformed_Sig_modulus_2)/F_SAMPLE
################################################################################
##spherical beamforming
def sample_MCMC(xant, yant, zant, signals, bound_function, X_0, n_steps=100, method='amplitude', step_size=1.):
    #Do Metropolis Hasting on the walker positions with the log likelyhood defined from the beamformed signal amplitude or power
    n_walkers = X_0.shape[0]
    if method not in {"amplitude", "power"}: raise NotImplementedError('Should be "amplitude" or "power"')
    compute_intensity = _compute_amp if method=='amplitude' else _compute_power
    
    current_walkers_pos = X_0.copy()
    all_walker_pos = np.zeros((n_steps, n_walkers, 3))
    all_walker_pos[0] = current_walkers_pos

    signal_intesity = np.zeros((n_steps, n_walkers))
    for i, walker_pos in enumerate(current_walkers_pos):
        _, beamformed_Sig_spherical = bm.spherical_beamforming(*walker_pos, xant, yant, zant, signals)
        spherical_beamformed_Sig_modulus_2 = beamformed_Sig_spherical[1,:]**2+beamformed_Sig_spherical[2,:]**2+beamformed_Sig_spherical[3,:]**2
        current_intensity = compute_intensity(spherical_beamformed_Sig_modulus_2)
        signal_intesity[0, i] = current_intensity
    current_walkers_intensity = signal_intesity[0].copy()
    for i in range(1, n_steps):
        candidate_walkers_pos = mcmc_candidate(current_walkers_pos, step_size=step_size, bound_function=bound_function)
    
        #Could be vectorized
        for w, walker_pos in enumerate(current_walkers_pos):
            _, beamformed_Sig_spherical = bm.spherical_beamforming(*candidate_walkers_pos[w], xant, yant, zant, signals)
            spherical_beamformed_Sig_modulus_2 = beamformed_Sig_spherical[1,:]**2+beamformed_Sig_spherical[2,:]**2+beamformed_Sig_spherical[3,:]**2
            new_intensity_w = compute_intensity(spherical_beamformed_Sig_modulus_2)

            acceptance_ratio = new_intensity_w / current_walkers_intensity[w]
            if acceptance_ratio <= np.random.rand():
                current_walkers_pos[w] = walker_pos
                current_walkers_intensity[w] = current_walkers_intensity[w]
            else:
                current_walkers_pos[w] = candidate_walkers_pos[w]
                current_walkers_intensity[w] = new_intensity_w
        signal_intesity[i, :] = current_walkers_intensity

        all_walker_pos[i] = current_walkers_pos
    return all_walker_pos, signal_intesity
        

def sample_temper(xant, yant, zant, signals, bound_function, X_0, n_steps=100, method='amplitude'):
    n_walkers = X_0.shape[0]

    current_walkers_pos = X_0.copy()

    all_walker_pos = np.zeros((n_steps, n_walkers, 3))
    signal_intesity = np.zeros((n_steps, n_walkers))
    if method not in {"amplitude", "power"}: raise NotImplementedError('Should be "amplitude" or "power"')
    compute_intensity = _compute_amp if method=='amplitude' else _compute_power
    
    for i in range(n_steps):
        all_walker_pos[i] = current_walkers_pos
        
        #Could be vectorized
        for w, walker_pos in enumerate(current_walkers_pos):
            _, beamformed_Sig_spherical = bm.spherical_beamforming(*walker_pos, xant, yant, zant, signals)
            spherical_beamformed_Sig_modulus_2 = beamformed_Sig_spherical[1,:]**2+beamformed_Sig_spherical[2,:]**2+beamformed_Sig_spherical[3,:]**2
            signal_intesity[i, w] = compute_intensity(spherical_beamformed_Sig_modulus_2)
        current_walkers_pos = temper_candidate(current_walkers_pos, signal_intesity[i], bound_function)
    return all_walker_pos, signal_intesity



def sample_random_grid(xant, yant, zant, signals, x_range_, y_range_, z_range_):
    _beamformed_maximum_map = np.zeros((len(x_range_), len(y_range_), len(z_range_)))

    # for i in progressbar(range(len(x_range_)), prefix="", size=60, out=sys.stdout):
    #
    #     for j in range(len(y_range_)):
    #
    #         for k in range(len(z_range_)):
    for i in progressbar(range(1000), prefix="", size=60, out=sys.stdout):

        i, j, k = np.random.randint(0, len(z_range_), size=3)
        #beamform for all positions
        _phased_Sig_spherical, beamformed_Sig_spherical = bm.spherical_beamforming(x_range_[i], y_range_[j], z_range_[k], xant, yant, zant, signals)
        _spherical_beamformed_Sig_modulus = np.sqrt(beamformed_Sig_spherical[1,:]**2+beamformed_Sig_spherical[2,:]**2+beamformed_Sig_spherical[3,:]**2)
        _beamformed_maximum_map[i, j, k] = np.amax(_spherical_beamformed_Sig_modulus)


    ##find the best direction == highest signal amplitude
    _best_rec = np.unravel_index(_beamformed_maximum_map.argmax(), _beamformed_maximum_map.shape)
    _best_xsrc, _best_ysrc, _best_zsrc = x_range_[_best_rec[0]], y_range_[_best_rec[1]], z_range_[_best_rec[2]]

    return _best_xsrc, _best_ysrc, _best_zsrc, _beamformed_maximum_map, _best_rec

