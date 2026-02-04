import sys
import numpy as np
import h5py
from scipy import interpolate
from scipy.signal import butter, lfilter, filtfilt

################################################################################
##Global paths
GP_AntennaModels_path = "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing/beamforming/AntennaModels/"
GP_NoisePath = "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing/beamforming/NoiseModels/"
################################################################################
##Functions

def time_jitter(time, jitter_std):

    # draw a random time offset, for each trace, in a gaussian distribution and add it to the time array
    # jitter_std should be in the same unit as time (usually ns)
    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=np.shape(time)[0])
    jitter = np.repeat(jitter, np.shape(time)[-1]).reshape(np.shape(time)[0],np.shape(time)[-1])

    return time + jitter

def interp(x,y,z):
    return np.interp(x,y,z,left=0,right=0)

def interpol_at_new_x(a_x, a_y, new_x):
    """
    Taken from grandlib
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)

def add_noise(noise_file, freqs_mhz, seed=None):

    noise = h5py.File(noise_file, "r")
    noise_vamp = np.array(noise["Vamp"])*1.e6  # [V] ->[muV]
    noise_freq = np.array(noise["freqs"]) #MHz

    nb_freq = len(freqs_mhz)
    v_amplitude = np.zeros((nb_freq, 3))
    v_amplitude[:, 0] = interpol_at_new_x(noise_freq, noise_vamp[:, 0], freqs_mhz)
    v_amplitude[:, 1] = interpol_at_new_x(noise_freq, noise_vamp[:, 1], freqs_mhz)
    v_amplitude[:, 2] = interpol_at_new_x(noise_freq, noise_vamp[:, 2], freqs_mhz)

    rng   = np.random.default_rng(seed)
    amp   = rng.normal(loc=0, scale=v_amplitude.T, size=(3, nb_freq))
    phase = 2 * np.pi * rng.random(size=(3, nb_freq))
    v_complex = np.abs(amp * nb_freq / 2) * np.exp(1j * phase)

    return v_complex

def get_Voc_filtered(time, Ex, Ey, Ez, theta, phi, fminMHz, fmaxMHz, AntennaModel, NoisePath, FILTER=False, NOISE=False):

    if AntennaModel == "GP300" :
        filesleff=[GP_AntennaModels_path+'GP300Antenna_SNarm_leff.npy',GP_AntennaModels_path+'GP300Antenna_EWarm_leff.npy',GP_AntennaModels_path+'GP300Antenna_Zarm_leff.npy']
        fmin_antmodel = 30 ; fmax_antmodel = 251 ; phi_bins = 361 ; theta_bins = 181 #/!\ hardcoded #depends on NEC tables
        GalacticFile = NoisePath+"galactic_noise_0LST_GP300.hdf5"
        #_GroundFile = NoisePath+"ground_noise_0LST_GP300.hdf5" #doesn't exist yet
    elif AntennaModel == "Rhombic" :
        filesleff=[GP_AntennaModels_path+'rhombicY_leff_gain2.npy',GP_AntennaModels_path+'rhombicX_leff_gain2.npy',GP_AntennaModels_path+'rhombicZ_leff_gain2.npy'] #X and Y are inverted in the file naming...
        fmin_antmodel = 25 ; fmax_antmodel = 251 ; phi_bins = 360 ; theta_bins = 181 #/!\ hardcoded #depends on NEC tables
        GalacticFile = NoisePath+"galactic_noise_0LST_Rhombic.hdf5"
        GroundFile = NoisePath+"ground_noise_0LST_Rhombic.hdf5"
    else :
        print("Wrong antenna model -> processing stoped")
        sys.exit()

    nbant=3 #3 polarisation : X, Y, Z #/!\ hardcoded
    simtsamp=1.e-9 #1ns sampling #/!\ hardcoded
    freqs=np.arange(fmin_antmodel,fmax_antmodel) #/!\ hardcoded

    Ex=Ex*1.e-6 ; Ey=Ey*1.e-6 ; Ez=Ez*1.e-6 #from muV to V #/!\ hardcoded
    simsize=len(time)
    inttheta=int(theta) ; intphi=int(phi)%360 #for lookup tables /!\ %360 is because some tables go from 0° to 359°

    fftfreqs=np.fft.rfftfreq(simsize,simtsamp)

    freq, theta, phi, ltheta, lphi, lthetaphase, lphiphase = (np.zeros((len(freqs),phi_bins,theta_bins,nbant)) for i in range(7))
    lthetacomplex, lphicomplex = (np.zeros((len(freqs),phi_bins,theta_bins,nbant),dtype='complex') for i in range(2))
    thislthetacomplex, thislphicomplex = (np.zeros((len(freqs),nbant),dtype='complex') for i in range(2))
    fftlthetacomplex, fftlphicomplex = (np.zeros((len(fftfreqs),nbant),dtype='complex') for i in range(2))

    for i in range(nbant):
        a=np.load(filesleff[i])
        freq[:,:,:,i]=np.reshape(a[0,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        theta[:,:,:,i]=np.reshape(a[3,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        phi[:,:,:,i]=np.reshape(a[4,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        ltheta[:,:,:,i]=np.reshape(a[5,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        lphi[:,:,:,i]=np.reshape(a[6,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        lthetaphase[:,:,:,i]=np.reshape(a[7,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]
        lphiphase[:,:,:,i]=np.reshape(a[8,:,:],(len(freqs),phi_bins,theta_bins))[:,:,0:]

        lthetacomplex[:,:,:,i]=ltheta[:,:,:,i]*np.exp(1j*lthetaphase[:,:,:,i]*np.pi/180)
        lphicomplex[:,:,:,i]=lphi[:,:,:,i]*np.exp(1j*lphiphase[:,:,:,i]*np.pi/180)

        thislthetacomplex[:,i]=lthetacomplex[:,:,:,i][(phi[:,:,:,i]==intphi) & (theta[:,:,:,i]==inttheta)]
        thislphicomplex[:,i]=lphicomplex[:,:,:,i][(phi[:,:,:,i]==intphi) & (theta[:,:,:,i]==inttheta)]
        fftlthetacomplex[:,i]=interp(fftfreqs/1e6,freqs,np.real(thislthetacomplex[:,i]))+1j*interp(fftfreqs/1e6,freqs,np.imag(thislthetacomplex[:,i]))
        fftlphicomplex[:,i]=interp(fftfreqs/1e6,freqs,np.real(thislphicomplex[:,i]))+1j*interp(fftfreqs/1e6,freqs,np.imag(thislphicomplex[:,i]))

    Etheta=Ex*(np.cos(theta*np.pi/180)*np.cos(phi*np.pi/180))+Ey*(np.cos(theta*np.pi/180)*np.sin(phi*np.pi/180))+Ez*(-np.sin(theta*np.pi/180))
    Ephi=Ex*(-np.sin(phi*np.pi/180))+Ey*(np.cos(phi*np.pi/180))

    EthetaFFT=np.fft.rfft(Etheta)
    EphiFFT=np.fft.rfft(Ephi)

    Voc=(fftlthetacomplex.T*EthetaFFT+fftlphicomplex.T*EphiFFT).T

    if NOISE==True:
        fft_galnoise = add_noise(GalacticFile, fftfreqs/1e6, seed=None)
        fft_grdnoise = np.zeros((3, len(fftfreqs))) #hack because no ground noise for GP300 yet, to be fixed
        if AntennaModel == 'Rhombic': fft_grdnoise = add_noise(GroundFile, fftfreqs/1e6, seed=None) #hack because no ground noise for GP300 yet, to be fixed
        Voc += (fft_galnoise.T+fft_grdnoise.T)/1.e6 #from muV to V

    if FILTER==True:
        sel_filt = np.where((fftfreqs/1e6<fminMHz)|(fftfreqs/1e6>fmaxMHz))[0] #/!\ outside fmin and fmax -> Voc = 0
        Voc[sel_filt] = 0

    return np.vstack((time.T[1:],np.fft.irfft(Voc, axis=0).T*1.e6)) #compute voltages in muV to be consistent with inputs


def process_signals(efield_, azim, zen, fminMHz, fmaxMHz, AntennaModel, NoisePath, FILTER=False, NOISE=False):

    Vx, Vy, Vz, time = ([] for i in range(4))
    for antid in range(len(efield_[0,:,0])):
        t, vx, vy, vz= get_Voc_filtered(efield_[0, antid], efield_[1, antid], efield_[2, antid], efield_[3, antid], zen, azim, fminMHz, fmaxMHz, AntennaModel, NoisePath, FILTER, NOISE)
        Vx.append(vx) ; Vy.append(vy) ; Vz.append(vz) ; time.append(t)

    return np.array([time, Vx, Vy, Vz])

def add_time_jitter(time, jitter_std):
    """
    draw a random time offset, for each trace, in a gaussian distribution and add it to the time array
    jitter_std should be in the same unit as time (usually ns)
    """

    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=np.shape(time)[0])
    jitter = np.repeat(jitter, np.shape(time)[-1]).reshape(np.shape(time)[0],np.shape(time)[-1])

    return time + jitter

def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')  # (order, [low, high], btype)
    return lfilter(b, a, data) #causal

def filter_Efields(efields, fmin, fmax):
    """
    time axis = ns
    fmin, fmax = MHz
    """
    if (fmax!=0) and (fmin!=0):
        dt =(efields[0,0,1] - efields[0,0,0])
        fs = 1/dt * 1e3  #MHz
        filt_Efields = np.copy(efields)
        filt_Efields[1:,:,:] = _butter_bandpass_filter(filt_Efields[1:,:,:], fmin, fmax, fs)
        return filt_Efields
    else:
        return efields

def process_signals_Efield(efields, azim, zen, fmin, fmax, noise_std, jitter_std):
    """
    basic version of process_signals without antenna responses
    noise is from random gaussian distribution
    fmin, fmax = MHz
    noise_std = muV/m
    jitter_std = ns
    """

    signals = filter_Efields(efields, fmin, fmax)

    if noise_std !=0:
        noise = np.random.normal(0, noise_std, size=np.shape(signals[1:,:,:]))
        noise_filtered = filter_Efields(np.vstack((signals[0:1,:,:], noise)), fmin, fmax)[1:,:,:]
        std_filtered = np.mean( np.std(noise_filtered[:,:,:], axis=-1) )
        noise_scaled = noise_filtered * (noise_std / std_filtered) 
        signals[1:,:,:] += noise_scaled

    if jitter_std !=0:
        signals[0,:,:] = add_time_jitter(signals[0,:,:], jitter_std)

    return signals

################################################################################
##Tests
# import read_sims_module as rm
#
# path_to_library = "/Users/decoene/Documents/Projects/HERON/Neutrino_Phasing/NewSims/TauDecay_Library/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates.npz"
# tau_events, antennas, efields, sttimes = rm.read_library(path_to_library)
#
#
# eventi = 0
# azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos = rm.get_event(eventi, tau_events, antennas, efields, sttimes)
#
# #fmin, fmax, noise_std, jitter_std = 50, 200, 22., 5.
# fmin, fmax, noise_std, jitter_std = 50, 200, 22, 5
#
# signals = process_signals_Efield(event_efield, azim, zen, fmin, fmax, noise_std, jitter_std)
#
# anti = 40
# fa, (a1x, a2x) = plt.subplots(1, 2)
# fa.suptitle(f"fmin={fmin:.1f} MHz - fmax={fmax:.1f} MHz \n noise={noise_std:.1f} muV/m, jitter={jitter_std:.1f} ns")
# a1x.set_xlabel(r"$\rm time\ (ns)$")
# a1x.set_ylabel(r"$\rm \vec{E}\ (\mu V/m)$")
# a2x.set_xlabel(r"$\rm time\ (ns)$")
# a2x.set_ylabel(r"$\rm \vec{E}_{filt,\,noise\,jitter}\ (\mu V/m)$")
#
# a1x.plot(event_efield[0,anti,:], event_efield[1,anti,:], label='Ex')
# a1x.plot(event_efield[0,anti,:], event_efield[2,anti,:], label='Ey')
# a1x.plot(event_efield[0,anti,:], event_efield[3,anti,:], label='Ez')
# a1x.legend()
#
# a2x.plot(signals[0,anti,:], signals[1,anti,:], label='Ex')
# a2x.plot(signals[0,anti,:], signals[2,anti,:], label='Ey')
# a2x.plot(signals[0,anti,:], signals[3,anti,:], label='Ez')
# a2x.legend()
#
# plt.show()
