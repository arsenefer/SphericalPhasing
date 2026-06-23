import numpy as np

def temper_candidate(current_walkers_pos, signal_intesity, bound_function, step_size=1., T=1.):
    intensity_factor = 2 * (1 + 1e3/T * signal_intesity/signal_intesity.max())**2
    step = np.random.randn(len(current_walkers_pos), 3) * step_size / intensity_factor[:,None]
    new_pos = current_walkers_pos + step
    #check if new position is in bounds 
    in_bound = bound_function(new_pos[:,0], new_pos[:,1], new_pos[:,2])
    while in_bound.sum() < len(current_walkers_pos):
        n_bad = (~in_bound).sum()
        step = np.random.randn(n_bad, 3) * step_size / intensity_factor[~in_bound,None]
        new_pos[~in_bound] = current_walkers_pos[~in_bound] + step
        in_bound = bound_function(new_pos[:,0], new_pos[:,1], new_pos[:,2])
    return new_pos

def mcmc_candidate(current_walkers_pos, step_size=1., bound_function=None):
    candidate_pos = current_walkers_pos + np.random.randn(len(current_walkers_pos), 3) * step_size
    if bound_function is not None:
        in_bound = bound_function(candidate_pos[:,0], candidate_pos[:,1], candidate_pos[:,2])
        while in_bound.sum() < len(current_walkers_pos):
            n_bad = (~in_bound).sum()
            step = np.random.randn(n_bad, 3) * step_size
            candidate_pos[~in_bound] = current_walkers_pos[~in_bound] + step
            in_bound = bound_function(candidate_pos[:,0], candidate_pos[:,1], candidate_pos[:,2])
    return candidate_pos