import sys
import numpy as np
import matplotlib.pyplot as plt
import beamforming_module.signal_module as sm
################################################################################
##Functions

def load_library(path_to_library_):

    library = np.load(path_to_library_, allow_pickle=True)

    return library


def read_library(path_to_library_, event_list=None):
    library = load_library(path_to_library_)
    if event_list is None:
        tau_events = library['tau_grid']
        antennas   = library['antenna_grid']
        efields    = library['efield']
        sttimes    = library['starttimes']
    else:
        efields    = library['efield'][event_list]
        sttimes    = library['starttimes'][event_list]
        tau_events = library['tau_grid'][event_list]
        antennas   = library['antenna_grid']
    return tau_events, antennas, efields, sttimes



def get_event(event_number_, tau_events, antennas, efields, sttimes, sampling_period=0.1):

    azim = np.copy(tau_events['phi'][event_number_])
    zen = 90-np.copy(tau_events['theta'][event_number_])
    en_nu = np.copy(tau_events['Enu'][event_number_])
    en_tau = np.copy(tau_events['E'][event_number_])
    tau_pos = np.copy(tau_events['position'][event_number_]) #X, Y, Z
    tau_alt = np.copy(tau_events['coordinates'][event_number_][2])
    xmax_pos = np.copy(tau_events['Xmax_position'][event_number_]) #X, Y, Z
    xmax_alt = np.copy(tau_events['Xmax_coordinates'][event_number_][2])
    event_efield = efields[event_number_,:,:,:]*1.e6 #in muV/m
    antenna_pos = np.copy(antennas['position'])
    antenna_alt = antennas['coordinates'][:,2]

    #Altitude shift to get sea level coordinates (first antenna at ground_altitude)
    tau_pos[2] += antenna_alt[0]
    xmax_pos[2] += antenna_alt[0]
    antenna_pos[:,2] += antenna_alt[0]

    t0s = sttimes[event_number_,:]
    duration = (event_efield.shape[-1]-1) * sampling_period

    time_samples = np.arange(0, duration + sampling_period, sampling_period)       # sampling in the library: duration in ns with sampling_period resolution

    time_trace   = np.array(np.tile(time_samples, (len(t0s),1)).T + t0s).T

    event_efield = np.array([time_trace, event_efield[:,0,:], event_efield[:,1,:], event_efield[:,2,:]])

    return azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos


def read_trace(eventi, tau_events, antennas, efields, sttimes, f_min, f_max, noise_std, jitter_std):
    (azim, zen, en_nu, 
     en_tau, tau_pos, xmax_pos, 
     event_efield, antenna_pos) = get_event(eventi, tau_events, antennas, efields, sttimes)
    xant, yant, zant = antenna_pos.T
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
################################################################################
# ##Tests
# path_to_library = "/Users/decoene/Documents/Projects/HERON/Neutrino_Phasing/NewSims/TauDecay_Library/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates.npz"
# tau_events, antennas, efields, sttimes = read_library(path_to_library)
# Nevent = 250
#
# for eventi in range(0,Nevent):
#
#     azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos = get_event(eventi, tau_events, antennas, efields, sttimes)
#     print(f"Event number {eventi:d} : neutrino energy = {en_nu:.1e}GeV")
#     print(f"Tau decay : azimuth = {azim:.1f}°, zenith = {zen:.1f}°, tau energy = {en_tau:.1e}GeV, position = ({tau_pos[0]:.1f}, {tau_pos[1]:.1f}, {tau_pos[2]:.1f}) m")
#     k_shower = np.array([np.cos(azim*np.pi/180)*np.sin(zen*np.pi/180), np.sin(azim*np.pi/180)*np.sin(zen*np.pi/180), np.cos(zen*np.pi/180)])
#
#     fa, (a1x, a2x) = plt.subplots(1,2)
#     a1x.set_xlabel(r"$\rm X-axis\ (km)$")
#     a1x.set_ylabel(r"$\rm Y-axis\ (km)$")
#     a2x.set_xlabel(r"$\rm X-axis\ (km)$")
#     a2x.set_ylabel(r"$\rm Z-axis\ (m)$")
#
#     distance_xy = 40
#     distance_xz = 150
#
#     a1x.scatter(antenna_pos[:,0]/1.e3, antenna_pos[:,1]/1.e3)
#     a1x.scatter(tau_pos[0]/1.e3, tau_pos[1]/1.e3, color='blue')
#     a1x.scatter(xmax_pos[0]/1.e3, xmax_pos[1]/1.e3, color='red')
#     a1x.plot([tau_pos[0]/1.e3, tau_pos[0]/1.e3+k_shower[0]*distance_xy], [tau_pos[1]/1.e3, tau_pos[1]/1.e3+k_shower[1]*distance_xy], color='k')
#
#     a2x.scatter(antenna_pos[:,0]/1.e3, antenna_pos[:,2]/1.e3)
#     a2x.scatter(tau_pos[0]/1.e3, tau_pos[2], color='blue')
#     a2x.scatter(xmax_pos[0]/1.e3, xmax_pos[1]/1.e3, color='red')
#     a2x.plot([tau_pos[0]/1.e3, tau_pos[0]/1.e3+k_shower[0]*distance_xz], [tau_pos[2], tau_pos[2]+k_shower[2]*distance_xz], color='k')
#
#     fb, bx = plt.subplots()
#     bx.set_xlabel(r"$\rm time\ (ns)$")
#     bx.set_ylabel(r"$\rm E-field\ (\mu V/m)$")
#
#     for anti in range(0, len(antenna_pos[:,0])):
#
#         bx.plot(event_efield[0, anti,:], event_efield[1, anti,:], linestyle='-')
#         bx.plot(event_efield[0, anti,:], event_efield[2, anti,:], linestyle='--')
#         bx.plot(event_efield[0, anti,:], event_efield[3, anti,:], linestyle=':')
#
#     plt.show()
