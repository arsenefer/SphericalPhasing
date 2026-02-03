import time
import sys
import numpy as np
import scipy.signal as ss
# sys.path.insert(0, '/Users/decoene/Documents/MyLibs_python/ZHAireS-Python')
# import AiresInfoFunctions as AiresInfo
from ..AiresInfoFunctions import (GetZHSEffectiveRefractionIndex, ZHSEffectiveRefractionIndexvect, ZHSEffectiveRefractionIndexvectTwice)
################################################################################
kc = 299792458. #m/s
kc_ns = 299792458.*1.e-9 #m/ns
kn = 1.0003
kcn_ns = kc_ns / kn

##phasing and beamforming functions
def signals_summing(phased_signals):

    _tstep = .1 #ns
    first_time = np.amin(phased_signals[0,:,:])    #First time of first antenna
    last_time = np.amax(phased_signals[0,:,:])     #Last time of last antenna
    _tarray = np.arange(first_time, last_time+_tstep, _tstep)
    _sumefx, _sumefy, _sumefz = np.zeros(len(_tarray)), np.zeros(len(_tarray)), np.zeros(len(_tarray))

    _start = np.int_((phased_signals[0,:,0] - first_time)/_tstep+_tstep)   #Start idx for each antenna
    _end = _start+phased_signals.shape[2] #Last idx for each antenna (length of traces)

    for anti in range(len(phased_signals[0,:,0])):
        _sumefx[_start[anti]:_end[anti]] += phased_signals[1,anti,:]
        _sumefy[_start[anti]:_end[anti]] += phased_signals[2,anti,:]
        _sumefz[_start[anti]:_end[anti]] += phased_signals[3,anti,:]

    return np.array([_tarray, _sumefx, _sumefy, _sumefz])


def spherical_phasing(xsrc, ysrc, zsrc, signals, xant, yant, zant):
    _ctdelays = np.sqrt((xant-xsrc)**2+(yant-ysrc)**2+(zant-zsrc)**2)

    ### This takes a long time. See if necessary to copy signals each time
    _phased_signals = np.copy(signals) #mitigate mutability issue

    # _neff = np.array([GetZHSEffectiveRefractionIndex(xsrc, ysrc, zsrc,xant=xant[i],yant=yant[i],zant=zant[i],ns=325,kr=-0.1218,stepsize = 20000) for i in range(len(xant))])
    _neff_vect = ZHSEffectiveRefractionIndexvect(np.array([xsrc, ysrc, zsrc]), np.array([xant, yant, zant]).T, ns=325, kr=-0.1218, stepsize = 20000)
    #6 times faster when vect

    _phased_signals[0,:,:] = (_phased_signals[0,:,:] - (_ctdelays/kc_ns*_neff_vect)[:,None]) #shift time axis of each antenna

    return _phased_signals


def spherical_beamforming(xsrc, ysrc, zsrc, xant, yant, zant, signals):
    _phased_signals = spherical_phasing(xsrc, ysrc, zsrc, signals, xant, yant, zant)
    _beamformed_signals = signals_summing(_phased_signals)
    return _phased_signals, _beamformed_signals


def _test_ref_index_vect(xsrc, ysrc, zsrc, xant, yant, zant):
    t_0 = time.time()
    for i in range(100):
        _neff = np.array([GetZHSEffectiveRefractionIndex(xsrc*100, ysrc*100, zsrc*100,xant=xant[i],yant=yant[i],zant=zant[i],ns=325,kr=-0.1218,stepsize = 20000) for i in range(len(xant))])
    t_1 = time.time()
    for i in range(100):
        _neff_vect = ZHSEffectiveRefractionIndexvect(np.array([xsrc*100, ysrc*100, zsrc*100]), np.array([xant, yant, zant]).T, ns=325,kr=-0.1218,stepsize = 20000)
    t_2 = time.time()
    Xs = np.array([[xsrc*100, ysrc*100, zsrc*100]]*100)
    _neff_vect_2 = ZHSEffectiveRefractionIndexvectTwice(Xs, np.array([xant, yant, zant]).T, ns=325,kr=-0.1218,stepsize = 20000)
    t_3 = time.time()
    print(f"Scalar refraction index time: {t_1 - t_0:.4f} s")
    print(f"Vectorised refraction index time: {t_2 - t_1:.4f} s")
    print(f"Vectorised twice refraction index time: {t_3 - t_2:.4f} s")
    print(f"Speed-up 1: {(t_1 - t_0)/(t_2 - t_1):.1f}x")
    print(f"Speed-up 2: {(t_1 - t_0)/(t_3 - t_2):.1f}x")
    assert np.allclose(_neff, _neff_vect), "Vectorised refraction index function does not match scalar version"
    assert np.allclose(_neff, _neff_vect_2[0]), "Vectorised refraction index function does not match scalar version"
    assert np.allclose(_neff_vect_2[0], _neff_vect_2[1]), "Vectorised refraction index function does not match scalar version"
    exit()