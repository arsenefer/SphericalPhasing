import numpy as np
import pandas as pd
import sys
sys.path.append('/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2')

import matplotlib.pyplot as plt
import beamforming_module.read_sims_module as rm
from beamforming_module.adf_weights import  ADF_parameters, shower_direction_vector
from beamforming_module.utils import Bn
import beamforming_module.beamforming_module_para as bm
import beamforming_module.sampling_module_para as sp

R2D = 180/np.pi

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

class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    # path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"
    path_to_library: str = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf'
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
    temp: float = 5000.  # Temperature for sampling
    x_max_method: str = 'max'  # Method for finding maximum
    n_best_walkers: int = 0  # Number of best walkers to consider
    sep_walkers: bool = False  # Separate walkers flag
    burn_in: int = 0  # Number of burn-in steps to discard

def plot_angular_error_histogram(rec, config, name_prefix, save=False):
    plt.hist(rec['angular_error_deg'], bins=100, range=(0,5));
    plt.xlabel('Angular error (degrees)')
    plt.ylabel('Number of events')
    plt.title('Distribution of Angular Errors in ADF Reconstruction')
    plt.grid()
    if save:
        plt.savefig(f'{config.save_path}/{name_prefix}angular_error_histogram.png', dpi=300)

def plot_rank_map(rec, config, name_prefix, rank, save=False):
    best_id = rec.sort_values('angular_error_deg').event_id.values[rank]
    map_best = np.load(f'results_adf/results_cc_F/{name_prefix}_{best_id:.0f}_intensity_map.npy')
    range_phi = np.linspace(-4, 4, 100)
    range_theta = np.linspace(-4, 4, 100)

    rec_theta = rec.sort_values('angular_error_deg').theta_error_deg.values[rank]
    rec_phi = rec.sort_values('angular_error_deg').phi_error_deg.values[rank]
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(range_phi, range_theta, map_best, shading='auto')
    ax.scatter(0,0, color='red', marker='x', label='True Direction')
    ax.scatter(rec_phi, rec_theta, color='blue', marker='o', label='Reconstructed Direction')
    ax.set_xlabel('Phi Error (degrees)')
    ax.set_ylabel('Theta Error (degrees)')
    ax.set_title('ADF Intensity Map for Best Reconstructed Event')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(c)
    if save:
        plt.savefig(f'{config.save_path}/{name_prefix}_event_map_rank.png', dpi=300)

def plot_best_map(rec, config, name_prefix, save=False):
    plot_rank_map(rec, config, name_prefix, rank=0, save=False)
    if save:
        plt.savefig(f'{config.save_path}/{name_prefix}_best_event_map.png', dpi=300)

def plot_event(signals, xant, yant, zant, tau_pos, xmax_pos, shower_dir, phi, theta, event_id):
    amps = np.max( np.linalg.norm(signals[1:], axis=0), axis=-1)
    # plt.scatter(xant/1e3, yant/1e3, c=amps, cmap='viridis', s=20)
    end_point = tau_pos + shower_dir*np.linalg.norm(tau_pos)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'Antenna Layout with Signal Amplitudes for Event {event_id}, $\\theta$={theta*R2D:.1f}°, $\\phi$={phi*R2D:.1f}°')

    # sca = ax[0].scatter(xant/1e3, yant/1e3, c=10*np.log10(amps), cmap='viridis', s=20)
    sca = ax[0].scatter(xant/1e3, yant/1e3, c=amps, cmap='viridis', s=20)
    ax[0].scatter(xmax_pos[0]/1e3, xmax_pos[1]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    ax[0].scatter(tau_pos[0]/1e3, tau_pos[1]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    ax[0].plot([tau_pos[0]/1e3, end_point[0]/1e3], [tau_pos[1]/1e3, end_point[1]/1e3], c='blue', linestyle='--', label='Shower Direction')
    ax[0].set_xlabel('X Antenna Position (km)')
    ax[0].set_ylabel('Y Antenna Position (km)')
    fig.colorbar(label='Max Signal Amplitude', ax=ax[0], mappable=sca)

    ax[1].scatter(xant/1e3, zant/1e3, c=amps, cmap='viridis', s=20)
    ax[1].scatter(xmax_pos[0]/1e3, xmax_pos[2]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    ax[1].scatter(tau_pos[0]/1e3, tau_pos[2]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    ax[1].plot([tau_pos[0]/1e3, end_point[0]/1e3], [tau_pos[2]/1e3, end_point[2]/1e3], c='blue', linestyle='--', label='Shower Direction')
    ax[1].set_xlabel('X Antenna Position (km)')
    ax[1].set_ylabel('Z Antenna Position (km)')
    fig.colorbar(label='Max Signal Amplitude', ax=ax[1], mappable=sca)

    plt.figure()    
    plt.scatter(yant/1e3, zant/1e3, c=amps, cmap='viridis', s=20)
    plt.scatter(xmax_pos[1]/1e3, xmax_pos[2]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    plt.scatter(tau_pos[1]/1e3, tau_pos[2]/1e3, c='red', marker='*', s=200, label='Xmax Position')
    plt.plot([tau_pos[1]/1e3, end_point[1]/1e3], [tau_pos[2]/1e3, end_point[2]/1e3], c='blue', linestyle='--', label='Shower Direction')
    plt.xlabel('Y Antenna Position (km)')
    plt.ylabel('Z Antenna Position (km)')
    plt.gca().set_aspect('equal')

def open_event(event_id, config, tau_events, antennas, efields, sttimes):

    (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos, 
         xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
                event_id, tau_events, antennas, efields, sttimes, config.f_min, config.f_max, config.noise_std, config.jitter_std
            )
    return event_id, tau_pos, xmax_pos, shower_dir, xant, yant, zant, signals, phi, theta


def plot_adf(theta, phi, xmax_pos, xant, yant, zant, signals):
    def smooth_adf(theta, phi, omega, omega_cr, Bn, eta, l_ant, delta_omega=0.1):
        K = shower_direction_vector(theta, phi)
    
        asym_coeff = -0.003*np.rad2deg(theta)+0.220

        asym = asym_coeff/np.sqrt(1. - np.dot(K,Bn)**2)

        adf = 1/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
        adf *= 1. + asym*np.cos(eta) # 
        return adf
    eta, omega, omega_cr, l_ant, adf = ADF_parameters(theta, phi, delta_omega=1,
                                                  Xants=np.stack((xant, yant, zant), axis=-1),
                                                  Xsource=xmax_pos)
    
    amps = np.max( np.linalg.norm(signals[1:], axis=0), axis=-1)
    mask = amps > (13*np.sqrt(3))*.0
    
    omega_smooth = np.linspace(-4, 4, 500)/R2D
    adf_smooth = smooth_adf(theta, phi, omega_smooth, omega_cr[0], Bn, 0, 1, delta_omega=1.)
    factor = amps[mask].max()/adf_smooth.max()*1
    
    omega_signed = omega * np.sign(eta)
    sorted_omega_signed = np.argsort(omega_signed[mask])
    
    plt.scatter(omega_signed[mask]*R2D, amps[mask], c='blue', s=20)
    plt.plot(omega_smooth*R2D, adf_smooth*factor+2, c='blue', linestyle='--', label='Sorted Amplitudes')
    plt.axvline(omega_cr[0]*R2D, color='red', linestyle='--', label='Critical Angle')
    plt.axvline(-omega_cr[0]*R2D, color='red', linestyle='--', label='Critical Angle')


def open_event_by_rank(rec, rank, config, tau_events, antennas, efields, sttimes):
    event_id = rec.sort_values('angular_error_deg').event_id.values[rank]
    return open_event(event_id, config, tau_events, antennas, efields, sttimes)

if __name__ == "__main__":
    config = Config()

    name_prefix = "LEevents" if "2e7" in config.path_to_library else "HEevents"

    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)

    rec = pd.read_csv(f'results_adf/results_cc_F/{name_prefix}_reconstruction_results.csv')

    event_id, tau_pos, xmax_pos, shower_dir, xant, yant, zant, signals, phi, theta = open_event_by_rank(rec, rank=1, config=config, tau_events=tau_events, antennas=antennas, efields=efields, sttimes=sttimes)
    plot_adf(theta, phi, xmax_pos, xant, yant, zant, signals)
    plt.show()