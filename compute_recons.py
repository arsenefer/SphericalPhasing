import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from dataclasses import dataclass
from time import time

# Importing custom modules
# import beamforming.sequential.beamforming_module as bm
import beamforming.read_sims_module as rm
import beamforming.signal_module as sm
import beamforming.sampling_module_para as sp
import beamforming.parameter_reconstruction as rec

from beamforming.sampler_init import (
    cubic_bound_function, 
    cubic_init, 
    sphere_bound_function, 
    sphere_init, 
    xyz_parameter_bounds, 
    flat_sphere_bound_function, 
    flat_sphere_init,
    create_regular_grid
)

R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/data/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results'
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

def save_samples(saving_path, walker_positions, beamformed_intensity_array, eventi, sampling, n_walkers, n_steps, step_size, bounds, amp_or_fluence, temp, **kwargs):
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
    directory = f"{saving_path}/AAAAAAAAAAAAAAAAAAAAAAAAAAAA_samples_sampling{sampling}_nwalkers{n_walkers}_nsteps{n_steps}_stepsize{step_size:.0f}_bounds{bounds}_intens{amp_or_fluence}_temp{temp:.0f}"
    os.makedirs(directory, exist_ok=True)
    np.savez(f"{directory}/event{eventi}.npz", 
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


def main(config: Config):
    """
    Main function to perform the reconstruction process.

    Parameters:
        config (Config): Configuration object containing all parameters.
    """
    name_prefix = "LEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = range(8, 10)  # List of events to process
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
        reference_antenna = np.argmax(signals[1:].max(axis=-1).sum(axis=0))

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
                xant, yant, zant, signals.copy(), bound_function=bound_function, X_0=X_0, 
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
                xant, yant, zant, signals.copy(), bound_function=bound_function, X_0=X_0, 
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
                xant, yant, zant, signals.copy(), walker_positions, 
                method=config.intens_method, T=config.temp, reference_antenna=reference_antenna
            )
            beamformed_intensity_array[np.isnan(beamformed_intensity_array)] = 0.
            beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array, T=None, threshold=2000.)

        elif config.sampling == "random":
            walker_positions = np.array([init_function(config.n_walkers, xmax_pos, config.r, config.thickness) for _ in range(config.n_steps)])
            walker_positions, beamformed_intensity_array = sp.presampled_beamforming(
                xant, yant, zant, signals.copy(), walker_positions, 
                method=config.intens_method, T=config.temp, reference_antenna=reference_antenna
            )
            beamformed_proba_array = rec.make_proba_array(beamformed_intensity_array, T=None, threshold=2000.)
        elif config.sampling == "3planes":
            walker_positions = sample_3planes(config.n_walkers, config.n_steps, xmax_pos, config.r, shower_dir)
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
                     config.temp, R=config.r, thickness=config.thickness if config.bounds == 'flat_sphere' else None)
        plot_recons_map(walker_positions[:, 45:55, :], 
                        beamformed_intensity_array[:, 45:55]**2 / beamformed_intensity_array[:, 45:55].mean(), 
                        X_0, xmax_pos, tau_pos, best_xsrc, best_ysrc, signals, (0, 0), burn_in=config.burn_in, 
                        shower_dir=shower_dir, shower_dir_rec=k_rec, xant=xant, yant=yant)

        # Compute distance from xmax
        vect_from_xmax = walker_positions.reshape(-1, 3) - xmax_pos[None, :]
        distance = np.linalg.norm(np.cross(vect_from_xmax, shower_dir[None, :]), axis=1)

        # Append results
        results.append({
            'eventi': eventi,
            'true_theta': np.arccos(shower_dir[2]),
            'true_phi': np.arctan2(shower_dir[1], shower_dir[0]),
            'theta_rec': theta_rec,
            'phi_rec': phi_rec,
            'best_xsrc': best_xsrc,
            'best_ysrc': best_ysrc,
            'best_zsrc': best_zsrc,
            "_tau_pos_x": tau_pos[0],
            "_tau_pos_y": tau_pos[1],
            "_tau_pos_z": tau_pos[2],
            "_xmax_pos_x": xmax_pos[0],
            "_xmax_pos_y": xmax_pos[1],
            "_xmax_pos_z": xmax_pos[2],
            'laterr': laterr,
            'longerr': longerr,
            'relerr': relerr,
            'err_theta': err_theta,
            'err_phi': err_phi,
            'err_angular': err_angular
        })

    # Save results to a CSV file
    results = pd.DataFrame(results)
    results.to_csv(f"{config.save_path}/samples_sampling{config.sampling}_nwalkers{config.n_walkers}_nsteps{config.n_steps}_stepsize{config.step_size:.0f}_bounds{config.bounds}_intens{config.intens_method}_temp{config.temp:.0f}/recons_base.csv", index=False)


if __name__ == "__main__":
    try:
        print("Starting the reconstruction process...")
        config = Config() 
        main(config)
    except KeyboardInterrupt:
        print('Exiting')