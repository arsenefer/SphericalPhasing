import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dataclasses import dataclass
from time import time
import uproot
from glob import glob

sys.path.append('/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')
# Importing custom modules
# import beamforming.sequential.beamforming_module as bm
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

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/ROOT_AND_TRACES/GROOT_DS/DC2.1rc4/ZHaireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0004/"
    save_path: str = './results_cr'
    sampling: str = 'random'
    n_walkers: int = 200
    n_steps: int = 600
    step_size: float = 1000.
    f_min: float = 0
    f_max: float = 5000
    noise_std: float = 0
    jitter_std: float = 0
    bounds: str = 'flat_sphere'  # Options: 'cubic', 'sphere', 'flat_sphere'
    r: float = 5.e3  # Radius for spherical bounds
    thickness: float = 4e3  # Thickness for flat spherical bounds
    intens_method: str = 'amplitude'  # Intensity calculation method
    temp: float = 5000.  # Temperature for sampling
    x_max_method: str = 'max'  # Method for finding maximum
    n_best_walkers: int = 0  # Number of best walkers to consider
    sep_walkers: bool = False  # Separate walkers flag
    burn_in: int = 0  # Number of burn-in steps to discard


def read_root(path_to_root_file):
    """
    Read the ROOT file and extract necessary data.

    Parameters:
        path_to_root_file (str): Path to the ROOT file.
    Returns:
        tuple: Extracted data including tau events, antennas, efields, and start times. 
    """
    tau_events, antennas, efields, sttimes = rm.read_library(path_to_root_file)
    return tau_events, antennas, efields, sttimes


def open_event_root(directory_to_roots, start=0, stop=10, L1_or_L0='0'):
    """
    Open the ROOT file containing the event data.

    Parameters
    ----------
    directory_to_roots : str
        The path to the directory containing the ROOT files.
    start : int, optional
        The starting index for reading entries. Default is 0.
    stop : int, optional
        The stopping index for reading entries. Default is None.
    L1_or_L0 : str, optional
        Specify whether to use L1 or L0 data. Default is '0'.

    Returns
    -------
    tuple
        - antenna_pos : ndarray
            The positions of the antennas.
        - meta_data : dict
            Metadata about the shower, including core position, zenith, azimuth, etc.
        - efield_data : dict
            The electric field time traces and associated data.
    """
    antenna_pos_file = sorted(glob(f'{directory_to_roots}/run_*_L0_*.root'))[0]
    shower_meta_data_files = sorted(glob(f'{directory_to_roots}/shower_*_L0_*.root'))
    efield_files = sorted(glob(f'{directory_to_roots}/efield_*_L{L1_or_L0}_*.root'))

    with uproot.open(antenna_pos_file) as f:
        antenna_pos = f['trun']['du_xyz'].array().to_numpy()[0]
    
    n_events = []
    for met in shower_meta_data_files:
        with uproot.open(met) as f:
            n_events.append(f['tshower'].num_entries)
    n_events = np.array(n_events)
    if stop is None:
        stop = np.sum(n_events)

    last_indices = np.cumsum(n_events)
    first_indices = last_indices - n_events
    mask_above_start = (last_indices > start)
    mask_below_stop = (first_indices < stop)
    overlap = np.where(mask_above_start & mask_below_stop)[0]
    zenith, azimuth, energy_primary, energy_em, xmax_grams, ptypes, event_numbers, efield_event_number = [np.array([]) for _ in range(8)]
    shower_core_pos, xmax_pos = [np.array([[]]).reshape(0, 3) for _ in range(2)]

    efield_trace, efield_du_ns, efield_du_s, efield_du_id, efield_du_pos, file_names = [], [], [], [], [], []
    for index_overlap in overlap:
        shower_meta_data_file = shower_meta_data_files[index_overlap]
        efield_file = efield_files[index_overlap]
        start_index = max(start, first_indices[index_overlap]) - first_indices[index_overlap]
        stop_index = min(stop, last_indices[index_overlap]) - first_indices[index_overlap]
        with uproot.open(shower_meta_data_file) as f:
            shower_meta_data = f['tshower']
            shower_core_pos = np.concatenate((shower_core_pos, shower_meta_data['shower_core_pos'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            zenith = np.concatenate((zenith, shower_meta_data['zenith'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180))
            azimuth = np.concatenate((azimuth, shower_meta_data['azimuth'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180))
            energy_primary = np.concatenate((energy_primary, shower_meta_data['energy_primary'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            energy_em = np.concatenate((energy_em, shower_meta_data['energy_em'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            xmax_grams = np.concatenate((xmax_grams, shower_meta_data['xmax_grams'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            xmax_pos = np.concatenate((xmax_pos, shower_meta_data['xmax_pos_shc'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            ptypes = np.concatenate((ptypes, shower_meta_data['primary_type'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            event_numbers = np.concatenate((event_numbers, shower_meta_data['event_number'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))

        with uproot.open(efield_file) as f:
            batch_du_ns = f['tefield']['du_nanoseconds'].array(
                entry_start=start_index, entry_stop=stop_index)
            batch_du_s = f['tefield']['du_seconds'].array(
                entry_start=start_index, entry_stop=stop_index)
            batch_du_id = f['tefield']['du_id'].array(
                entry_start=start_index, entry_stop=stop_index)
            
            efield_trace += [traces.to_numpy() for traces in f['tefield']['trace'].array(
                entry_start=start_index, entry_stop=stop_index)]
            efield_du_ns += [du_ns.to_numpy() for du_ns in batch_du_ns]
            efield_du_s += [du_s.to_numpy() for du_s in batch_du_s]
            efield_du_id += [du_id.to_numpy() for du_id in batch_du_id]
            efield_du_pos += [antenna_pos[du_id.to_numpy()] for du_id in batch_du_id]

            efield_event_number = np.concatenate((efield_event_number, f['tefield']['event_number'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
        file_names += [efield_file] * (stop_index - start_index)

    # xmax_pos = xmax_pos + shower_core_pos - np.array([[0, 0, altitude]])
    
    
    ptypes_int = np.ones_like(ptypes, dtype=int) 
    ptypes_int[ptypes == 'Fe^56'] = 56
    xmax_pos = xmax_pos + shower_core_pos
    k_p = shower_core_pos - xmax_pos
    k = k_p / np.linalg.norm(k_p, axis=1, keepdims=True)
    th_p = np.arccos(-k[:, 2])
    assert np.allclose(th_p * R2D, zenith * R2D, atol=1e-4, rtol=1e-4), f"{th_p*R2D} vs {zenith*R2D}"
    assert np.allclose(shower_core_pos[:,2], antenna_pos[:,2].mean(), atol=1e2, rtol=0), f"{shower_core_pos[:,2].mean()} vs {antenna_pos[:,2].mean()}"
    showers_params = (azimuth, zenith, energy_primary, energy_em, shower_core_pos, xmax_pos)

    fs = 2e9 # 2 GHz
    duration = efield_trace[0].shape[-1] / fs * 1e9 # in nanoseconds
    time_arrays = [np.repeat(np.linspace(0, duration, trace.shape[-1], endpoint=False)[None,:], trace.shape[0], axis=0) for trace in efield_trace]  # in nanoseconds
    time_arrays = [time_array + (du_s - du_s.min())[:, None]*1e9 + du_ns[:, None] for time_array, du_s, du_ns in zip(time_arrays, efield_du_s, efield_du_ns)]  # in ns
    time_arrays = [time_array - time_array.min() for time_array in time_arrays]  # in ns
    efields = [np.concatenate((time_array[None, :, :], np.swapaxes(trace, 0, 1)), axis=0) for time_array, trace in zip(time_arrays, efield_trace)]
    # plt.scatter(antenna_pos[:,0], antenna_pos[:,1], c='gray', s=2)
    # plt.scatter(efield_du_pos[2][:,0], efield_du_pos[2][:,1], c='red', s=5)
    # plt.show()   
    # plt.scatter(antenna_pos[:,0], antenna_pos[:,1], c='gray', s=2)
    # plt.scatter(efield_du_pos[3][:,0], efield_du_pos[3][:,1], c='red', s=5)
    # plt.show()
    return showers_params, efield_du_pos, efields, None

def read_trace(event_idx, tau_events, all_antenna_pos, efields, sttimes, f_min, f_max, noise_std, jitter_std):
    azim, zen, en_nu, en_tau, tau_pos, xmax_pos = tau_events[0][event_idx], tau_events[1][event_idx], tau_events[2][event_idx], tau_events[3][event_idx], tau_events[4][event_idx], tau_events[5][event_idx]
    event_efield, antenna_pos = efields[event_idx], all_antenna_pos[event_idx]
    xant, yant, zant = antenna_pos[:, 0], antenna_pos[:, 1], antenna_pos[:, 2]

    signals =  sm.process_signals_Efield(event_efield, azim, zen, f_min, f_max, noise_std, jitter_std)

    tauxXmax_dist = np.linalg.norm(xmax_pos-tau_pos)
    shower_dir = (xmax_pos - tau_pos)/tauxXmax_dist
    theta = np.arccos(-shower_dir[2])
    phi = np.arctan2(-shower_dir[1], -shower_dir[0])

    return (azim, zen, en_nu, 
            en_tau, tau_pos, xmax_pos, 
            event_efield, antenna_pos, 
            xant, yant, zant, 
            signals, phi, theta, shower_dir)

def plot_recons_map(walker_positions, beamformed_intensity_array, X_0, xmax_pos, tau_pos, best_xsrc, best_ysrc, 
                    signals, rec_index, burn_in=50, shower_dir=None, shower_dir_rec=None, xant=None, yant=None):
    """
    Plot the reconstruction map and signal traces.

    Parameters:
        walker_positions (ndarray): Positions of the walkers.
        beamformed_intensity_array (ndarray): Beamformed intensity values.
        X_0 (ndarray): Initial positions of the walkers.
        xmax_pos (ndarray): Position of the maximum.
        tau_pos (ndarray): Tau position.
        best_xsrc, best_ysrc (float): Best source positions.
        signals (ndarray): Signal data.
        rec_index (tuple): Reconstruction index.
        burn_in (int): Number of burn-in steps.
        shower_dir (ndarray): True shower direction.
        shower_dir_rec (ndarray): Reconstructed shower direction.
        xant, yant (ndarray): Antenna positions.
    """
    sampling_period = signals[0, 0, 1] - signals[0, 0, 0]
    walker_positions_after_burn = walker_positions[burn_in:, :, :]
    beamformed_intensity_array_after_burn = beamformed_intensity_array[burn_in:, :]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot of walker positions
    ax[0].scatter(walker_positions_after_burn[:, :, 0].flatten(), 
                  walker_positions_after_burn[:, :, 1].flatten(), 
                  c=beamformed_intensity_array_after_burn.flatten(),
                  alpha=(beamformed_intensity_array_after_burn.flatten() / beamformed_intensity_array_after_burn.max()))
    ax[0].scatter(X_0[:, 0], X_0[:, 1], s=10)
    ax[0].scatter(xmax_pos[0], xmax_pos[1], marker='x', c='k', s=50, zorder=100000)
    ax[0].scatter(tau_pos[0], tau_pos[1], marker='x', c='k', s=50, zorder=100000)
    ax[0].scatter(best_xsrc, best_ysrc, marker='x', c='r')
    ax[0].scatter(xant, yant, marker='^', c='k', s=20)
    ax[0].set_aspect('equal', adjustable='datalim')

    # Add arrows for true and reconstructed shower directions
    if shower_dir is not None:
        ax[0].arrow(xmax_pos[0], xmax_pos[1], shower_dir[0] * 3000, shower_dir[1] * 3000, width=10, head_width=300, color='g')
    if shower_dir_rec is not None:
        ax[0].arrow(xmax_pos[0], xmax_pos[1], shower_dir_rec[0] * 3000, shower_dir_rec[1] * 3000, width=10, head_width=300, color='r')

    # Plot signal traces
    for i_s in range(signals.shape[1]):
        ax[1].plot(signals[0, i_s, :], signals[1, i_s, :])

    # Plot summed signals
    summed = dict()
    for i_s in range(signals.shape[1]):
        t_s = np.arange(signals.shape[-1]) * sampling_period + beamformed_intensity_array[rec_index]
        ax[2].plot(t_s, signals[1, i_s, :])
        for i in range(signals.shape[-1]):
            t = t_s[i]
            if t in summed:
                summed[t] += signals[1, i_s, i]
            else:
                summed[t] = signals[1, i_s, i]

    ax[2].plot(summed.keys(), summed.values())
    plt.show()


def save_samples(saving_path, walker_positions, beamformed_intensity_array, eventi, sampling, n_walkers, n_steps, step_size, bounds, amp_or_fluence, temp, prefix, **kwargs):
    """
    Save the sampled walker positions and beamformed intensity array to a file.

    Parameters:
        saving_path (str): Directory to save the results.
        walker_positions (ndarray): Positions of the walkers.
        beamformed_intensity_array (ndarray): Beamformed intensity values.
        eventi (int): Event index.
        sampling (str): Sampling method.
        n_walkers (int): Number of walkers.
        n_steps (int): Number of steps.
        step_size (float): Step size.
        bounds (str): Bounds type.
        amp_or_fluence (str): Intensity calculation method.
        temp (float): Temperature for sampling.
        **kwargs: Additional parameters to save.
    """
    directory = f"{saving_path}/samples_thresh10_sampling{sampling}_nwalkers{n_walkers}_nsteps{n_steps}_stepsize{step_size:.0f}_bounds{bounds}_intens{amp_or_fluence}_temp{temp:.0f}"
    os.makedirs(directory, exist_ok=True)
    np.savez(f"{directory}/{prefix}_event{eventi}.npz", 
             walker_positions=walker_positions, 
             beamformed_intensity_array=beamformed_intensity_array, 
             eventi=eventi, 
             sampling=sampling, 
             n_walkers=n_walkers, 
             n_steps=n_steps, 
             bounds=bounds, 
             amp_or_fluence=amp_or_fluence, 
             temp=temp,
             **kwargs)

def plot_traces(signals):
    """
    Plot the signal traces.

    Parameters:
        signals (ndarray): Signal data.
    """
    fig, axs = plt.subplots(5, 1, figsize=(6, 6), sharex=True, sharey=True)
    amps = np.linalg.norm(signals[1:], axis=0).max(axis=-1)
    sorted_idx = np.argsort(amps)[::-1]
    signalss = signals[:, sorted_idx, :]
    for i in range(5):
        pl1, = axs[i].plot(signalss[0, i, 1080:1120] - signalss[0,i,0], signalss[1, i, 1080:1120], label=f'E field X')
        pl2, = axs[i].plot(signalss[0, i, 1080:1120] - signalss[0,i,0], signalss[2, i, 1080:1120], label=f'E field Y')
        pl3, = axs[i].plot(signalss[0, i, 1080:1120] - signalss[0,i,0], signalss[3, i, 1080:1120], label=f'E field Z')

        axs[i].set_ylabel(f'Ant. {i+1}')
        axs[i].grid()
    axs[-1].set_xlabel('Time [ns]')
    fig.supylabel('E. field amp. [µV/m]')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.legend(loc='lower center', handles=[pl1, pl2, pl3], ncol=3, bbox_to_anchor=(.55, 0))
    plt.show()

def main(config: Config):
    """
    Main function to perform the reconstruction process.

    Parameters:
        config (Config): Configuration object containing all parameters.
    """
    name_prefix = "LEevents_"
    start=2
    stopstop = 10
    Eventlist = range(start, stopstop)  # List of events to process
    tau_events, all_antenna_pos, efields, sttimes = open_event_root(config.path_to_library, start=start, stop=stopstop)
    Nevents = len(Eventlist)

    results = []
    t_0 = time()

    for i, eventi in enumerate(Eventlist):
        np.random.seed(i)  # Set random seed for reproducibility

        # Read event data
        (azim, zen, en_nu, 
            en_tau, tau_pos, xmax_pos, 
            event_efield, antenna_pos, 
            xant, yant, zant, 
            signals, phi, theta, shower_dir) = read_trace(
            i, tau_events, all_antenna_pos, efields, sttimes, config.f_min, config.f_max, config.noise_std, config.jitter_std
        )
        plot_traces(signals)
        exit()
        amps = np.linalg.norm(signals[1:], axis=0).max(axis=-1)
        keep = (amps > 10) # antennas with signal > 50 uV
        if not (keep.sum() > 50 and zen*R2D < 60): # More than 60 antennas with signal > 50 uV
            print(f"Event number {eventi:d} : skipped due to low signal. idx = {i}")
            continue
        else:
            print("Das good")

        for side in ['all', 'left', 'right', 'top', 'bottom']:
            keep = (amps > 10) # antennas with signal > 50 uV
            if side == 'all':
                is_to_the_side = np.ones(len(antenna_pos), dtype=bool)
            elif side =='left':
                q = shower_dir[0]*(antenna_pos - tau_pos)[:,1] - shower_dir[1]*(antenna_pos - tau_pos)[:,0]
                is_to_the_side = q >= np.quantile(q[keep], .8)
            elif side =='right':
                q = shower_dir[0]*(antenna_pos - tau_pos)[:,1] - shower_dir[1]*(antenna_pos - tau_pos)[:,0]
                is_to_the_side = q <= np.quantile(q[keep], .2)
            elif side=='top':
                q = shower_dir[0]*(antenna_pos - tau_pos)[:,0] + shower_dir[1]*(antenna_pos - tau_pos)[:,1]
                is_to_the_side = q >= np.quantile(q[keep], .8)
            elif side=='bottom':
                q = shower_dir[0]*(antenna_pos - tau_pos)[:,0] + shower_dir[1]*(antenna_pos - tau_pos)[:,1]
                is_to_the_side = q <= np.quantile(q[keep], .2)
            # fig,axs = plt.subplots(2,2, figsize=(10,10))
            # axs = axs.flatten()
            # for ax in axs:
            #     ax.scatter(antenna_pos[keep,0], antenna_pos[keep,1], c='gray', s=2)
            #     ax.scatter(tau_pos[0], tau_pos[1], marker='x', c='k', s=50, zorder=100000)
            # q = shower_dir[0]*(antenna_pos - tau_pos)[:,1] - shower_dir[1]*(antenna_pos - tau_pos)[:,0]
            # sca = axs[0].scatter(antenna_pos[keep,0], antenna_pos[keep,1], c=q[keep], s=10)
            # plt.colorbar(sca, ax=axs[0], label='Left-Right separation')
            # sca = axs[2].scatter(antenna_pos[keep,0], antenna_pos[keep,1], c=(q > np.median(q[keep])).astype(float)[keep], s=10)
            # plt.colorbar(sca, ax=axs[2], label='Left-Right separation')

            # q = shower_dir[0]*(antenna_pos - tau_pos)[:,0] + shower_dir[1]*(antenna_pos - tau_pos)[:,1]
            # sca = axs[1].scatter(antenna_pos[keep,0], antenna_pos[keep,1], c=q[keep], s=10)
            # plt.colorbar(sca, ax=axs[1], label='Top-Bottom separation')
            # sca = axs[3].scatter(antenna_pos[keep,0], antenna_pos[keep,1], c=(q > np.median(q[keep])).astype(float)[keep], s=10)
            # plt.colorbar(sca, ax=axs[3], label='Top-Bottom separation')
            # axs[2].scatter(antenna_pos[keep & is_to_the_side,0], antenna_pos[keep & is_to_the_side,1], marker='x', c='r', s=10)
            # axs[3].scatter(antenna_pos[keep & is_to_the_side,0], antenna_pos[keep & is_to_the_side,1], marker='x', c='r', s=10)
            # plt.show()
            keepside = keep & is_to_the_side
            signals_side = signals[:, keepside, :]
            xant_side, yant_side, zant_side = xant[keepside], yant[keepside], zant[keepside]
            reference_antenna = np.argmax(signals_side[1:].max(axis=-1).sum(axis=0))
            # Define bounds and initialize walkers based on the configuration
            if config.bounds == 'cubic':
                x_bounds, y_bounds, z_bounds = xyz_parameter_bounds(tau_pos, phi, theta)
                bound_function = cubic_bound_function(x_bounds, y_bounds, z_bounds)
                X_0 = cubic_init(config.n_walkers, x_bounds, y_bounds, z_bounds)
                init_function = cubic_init
                step_vect = np.array((1., 1., 1.)) * config.step_size
                Xrange = x_bounds
                Yrange = y_bounds
                Zrange = z_bounds

            elif config.bounds == 'sphere':
                bound_function = sphere_bound_function(xmax_pos, radius=config.r)
                X_0 = sphere_init(config.n_walkers, xmax_pos, config.r)
                init_function = sphere_init
                step_vect = np.array((1., 1., 1.)) * config.step_size
                Xrange = (xmax_pos[0] - config.r, xmax_pos[0] + config.r)
                Yrange = (xmax_pos[1] - config.r, xmax_pos[1] + config.r)
                Zrange = (xmax_pos[2] - config.r, xmax_pos[2] + config.r)

            elif config.bounds == 'flat_sphere':
                bound_function = flat_sphere_bound_function(xmax_pos, radius=config.r, thickness=config.thickness)
                X_0 = flat_sphere_init(config.n_walkers, xmax_pos, config.r, config.thickness)
                init_function = flat_sphere_init
                step_vect = np.array((1., 1., config.thickness / config.r)) * config.step_size
                Xrange = (xmax_pos[0] - config.r, xmax_pos[0] + config.r)
                Yrange = (xmax_pos[1] - config.r, xmax_pos[1] + config.r)
                Zrange = (xmax_pos[2] - config.thickness, xmax_pos[2] + config.thickness)
            else:
                raise NotImplementedError(f"Bounds should be either 'cubic', 'sphere', or 'flat_sphere'. Not {config.bounds}")

            # Sampling based on the selected method
            if config.sampling == "parallel_tempering":
                print(f"Event number {eventi:d} : neutrino energy = {en_nu:.1e}GeV")
                walker_positions, beamformed_intensity_array = sp.sample_temper(
                    xant_side, yant_side, zant_side, signals_side.copy(), bound_function=bound_function, X_0=X_0, 
                    n_steps=config.n_steps, method=config.intens_method, step_size=step_vect, T=config.temp,
                    reference_antenna=reference_antenna
                )
                beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array)
                k_rec, (theta_rec, phi_rec) = rec.direction_recons(
                    walker_positions, beamformed_proba_array, 
                    n_walkers_to_consider=config.n_best_walkers, sep_walkers=config.sep_walkers, 
                    burn_in=0, weighted=True
                )

            elif config.sampling == "metrop_hasting":
                walker_positions, beamformed_intensity_array = sp.sample_MCMC(
                    xant_side, yant_side, zant_side, signals_side.copy(), bound_function=bound_function, X_0=X_0, 
                    n_steps=config.n_steps, method=config.intens_method, step_size=step_vect, T=config.temp,
                    reference_antenna=reference_antenna
                )
                best_xsrc, best_ysrc, best_zsrc, rec_index = rec.max_recons(walker_positions, beamformed_intensity_array)
                k_rec, (theta_rec, phi_rec) = None, (None, None)
                beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array, T=config.temp, threshold=0.)

            elif config.sampling == "cubic_grid":
                X_0 = X_0 * np.nan
                walker_positions = create_regular_grid(config.n_walkers, config.n_steps, Xrange, Yrange, Zrange, bound_function)
                walker_positions, beamformed_intensity_array = sp.presampled_beamforming(
                    xant_side, yant_side, zant_side, signals_side.copy(), walker_positions, 
                    method=config.intens_method, T=config.temp, reference_antenna=reference_antenna
                )
                beamformed_intensity_array[np.isnan(beamformed_intensity_array)] = 0.
                beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array, T=None, threshold=2000.)

            elif config.sampling == "random":
                walker_positions = np.array([init_function(config.n_walkers, xmax_pos, config.r, config.thickness) for _ in range(config.n_steps)])
                walker_positions, beamformed_intensity_array = sp.presampled_beamforming(
                    xant_side, yant_side, zant_side, signals_side.copy(), walker_positions, 
                    method=config.intens_method, T=config.temp, reference_antenna=reference_antenna
                )
                beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array, T=None, threshold=2000.)
            else:
                raise ValueError("Invalid sampling method.")

            # Reconstruction and error computation
            best_xsrc, best_ysrc, best_zsrc, rec_index = rec.max_recons(walker_positions, beamformed_intensity_array)
            rec_pos, laterr, longerr, relerr = rec.compute_errors_source(best_xsrc, best_ysrc, best_zsrc, xmax_pos, tau_pos, shower_dir)
            k_rec, (theta_rec, phi_rec) = None, (None, None)

            if phi_rec is not None:
                err_theta = np.abs(theta_rec - theta)
                err_phi = np.abs(phi_rec - phi)
                err_angular = rec.get_angular_distance(phi, phi_rec, theta, theta_rec)
            else:
                err_theta = None
                err_phi = None
                err_angular = None

            # Save results and plot reconstruction map
            save_samples(config.save_path, walker_positions, beamformed_intensity_array, eventi, config.sampling, 
                        config.n_walkers, config.n_steps, config.step_size, config.bounds, config.intens_method, 
                        config.temp, R=config.r, thickness=config.thickness if config.bounds == 'flat_sphere' else None, prefix=side)
            # plot_recons_map(walker_positions[:, :, :], 
            #                 beamformed_intensity_array[:, :]**2 / beamformed_intensity_array[:, :].mean(), 
            #                 X_0, xmax_pos, tau_pos, best_xsrc, best_ysrc, signals_side, (0, 0), burn_in=config.burn_in, 
            #                 shower_dir=shower_dir, shower_dir_rec=k_rec, xant=xant_side, yant=yant_side)

            # Compute distance from xmax
            vect_from_xmax = walker_positions.reshape(-1, 3) - xmax_pos[None, :]
            distance = np.linalg.norm(np.cross(vect_from_xmax, shower_dir[None, :]), axis=1)

            # Append results
            # results.append({
            #     'eventi': eventi,
            #     'true_theta': np.arccos(shower_dir[2]),
            #     'true_phi': np.arctan2(shower_dir[1], shower_dir[0]),
            #     'theta_rec': theta_rec,
            #     'phi_rec': phi_rec,
            #     'best_xsrc': best_xsrc,
            #     'best_ysrc': best_ysrc,
            #     'best_zsrc': best_zsrc,
            #     "_tau_pos_x": tau_pos[0],
            #     "_tau_pos_y": tau_pos[1],
            #     "_tau_pos_z": tau_pos[2],
            #     "_xmax_pos_x": xmax_pos[0],
            #     "_xmax_pos_y": xmax_pos[1],
            #     "_xmax_pos_z": xmax_pos[2],
            #     'laterr': laterr,
            #     'longerr': longerr,
            #     'relerr': relerr,
            #     'err_theta': err_theta,
            #     'err_phi': err_phi,
            #     'err_angular': err_angular
            # })

        # Save results to a CSV file
        # results = pd.DataFrame(results)
        # results.to_csv(f"{config.save_path}/samples_sampling{config.sampling}_nwalkers{config.n_walkers}_nsteps{config.n_steps}_stepsize{config.step_size:.0f}_bounds{config.bounds}_intens{config.intens_method}_temp{config.temp:.0f}/{side}_recons_base.csv", index=False)


if __name__ == "__main__":
    try:
        print("Starting the reconstruction process...")
        config = Config() 
        main(config)
    except KeyboardInterrupt:
        print('Exiting')