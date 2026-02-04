import time
import sys
from line_profiler import profile
import numpy as np
import scipy.signal as ss
from .AiresInfoFunctions import (GetZHSEffectiveRefractionIndex, ZHSEffectiveRefractionIndexvect, ZHSEffectiveRefractionIndexvectTwice)
# Optional: use numba to JIT the inner accumulation loop (much faster than pure Python loop for moderate sizes)
from numba import njit, prange

################################################################################
kc = 299792458. #m/s
kc_ns = 299792458.*1.e-9 #m/ns
kn = 1.0003
kcn_ns = kc_ns / kn
################################################################################

##phasing and beamforming functions
def signals_summing(signals, shifted_times):
    """
        Compute the summed signal from different sources
        inputs:
            signals_ (np.ndarray): signals from simulations shape : (4, n_antennas, trance_len)
            shifted_times_ (np.ndarray): times of the begining of signals from every source. shape (n_sources, n_antennas)
        outputs:
            list of summed signal. list of np.ndarray of shape : (4, len_out_trace)
    """
    _tstep = signals[0,0,1] - signals[0,0,0] #ns
    trace_len = signals.shape[-1]
    n_sources, n_antennas = shifted_times.shape
    trace_duration = trace_len * _tstep
    list_out = []
    for source_times in shifted_times:
        first_time = np.min(source_times)    #First time of first antenna per sources shape (n_sources)
        last_time = np.max(source_times) + trace_duration #Last time of last antenna
        _tarray = np.arange(first_time, last_time+_tstep, _tstep)
        _sumef = np.zeros((4, len(_tarray)))
        _sumef[0] = _tarray


        _start_idx = np.int_((source_times - first_time)/_tstep+_tstep)   #Start idx for each antenna
        _end_idx = _start_idx+trace_len #Last idx for each antenna (length of traces)
    

        for anti in range(len(source_times)):
            _sumef[1, _start_idx[anti]:_end_idx[anti]] += signals[1,anti,:]
            _sumef[2, _start_idx[anti]:_end_idx[anti]] += signals[2,anti,:]
            _sumef[3, _start_idx[anti]:_end_idx[anti]] += signals[3,anti,:]
        
        list_out.append(_sumef)
    return list_out

##phasing and beamforming functions
def signals_summing_adf(signals, shifted_times, weights=None):
    """
        Compute the summed signal from different sources
        inputs:
            signals_ (np.ndarray): signals from simulations shape : (4, n_antennas, trance_len)
            shifted_times_ (np.ndarray): times of the begining of signals from every source. shape (n_sources, n_antennas)
        outputs:
            list of summed signal. list of np.ndarray of shape : (4, len_out_trace)
    """
    if weights is None:
        weights = np.ones(shifted_times.shape)
    _tstep = signals[0,0,1] - signals[0,0,0] #ns
    trace_len = signals.shape[-1]
    n_sources, n_antennas = shifted_times.shape
    trace_duration = trace_len * _tstep
    list_out = []
    for source_idx, source_times in enumerate(shifted_times):
        first_time = np.min(source_times)    #First time of first antenna per sources shape (n_sources)
        last_time = np.max(source_times) + trace_duration #Last time of last antenna
        _tarray = np.arange(first_time, last_time+_tstep, _tstep)
        _sumef = np.zeros((4, len(_tarray)))
        _sumef[0] = _tarray


        _start_idx = np.int_((source_times - first_time)/_tstep+_tstep)   #Start idx for each antenna
        _end_idx = _start_idx+trace_len #Last idx for each antenna (length of traces)
    

        for anti in range(len(source_times)):
            _sumef[1, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[1,anti,:]
            _sumef[2, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[2,anti,:]
            _sumef[3, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[3,anti,:]
        
        list_out.append(_sumef)
    return np.array(list_out)


def signals_summing_ref(signals, shifted_times, reference_antenna=0):
    """
        Compute the summed signal from different sources
        inputs:
            signals_ (np.ndarray): signals from simulations shape : (4, n_antennas, trance_len)
            shifted_times_ (np.ndarray): times of the begining of signals from every source. shape (n_sources, n_antennas)
        outputs:
            list of summed signal. list of np.ndarray of shape : (4, len_out_trace)
    """
    _tstep = signals[0,0,1] - signals[0,0,0] #ns
    trace_len = signals.shape[-1]
    n_sources, n_antennas = shifted_times.shape
    trace_duration = trace_len * _tstep
    output = np.zeros((n_sources, 4, trace_len))
    output[:,0,:] = np.arange(trace_len)[None,:] * _tstep + shifted_times[:,reference_antenna,None]
    for source_idx, source_times in enumerate(shifted_times):
        _sumef = np.zeros((3, trace_len))
        ref_time = source_times[reference_antenna]

        _start_idx = np.int_((source_times - ref_time)/_tstep)   #Start idx for each antenna
        _start_idx = np.clip(_start_idx, 0, trace_len-1)

        _end_idx = _start_idx+trace_len #Last idx for each antenna (length of traces)
        _end_idx = np.clip(_end_idx, 0, trace_len)

        for anti in range(n_antennas):
            _sumef[0, _start_idx[anti]:_end_idx[anti]] += signals[1,anti,:_end_idx[anti]-_start_idx[anti]]
            _sumef[1, _start_idx[anti]:_end_idx[anti]] += signals[2,anti,:_end_idx[anti]-_start_idx[anti]]
            _sumef[2, _start_idx[anti]:_end_idx[anti]] += signals[3,anti,:_end_idx[anti]-_start_idx[anti]]
        
        output[source_idx, 1:4,:] = _sumef
    return output

def signals_summing_ref_adf(signals, shifted_times, weights=None, reference_antenna=0):
    """
        Compute the summed signal from different sources
        inputs:
            signals_ (np.ndarray): signals from simulations shape : (4, n_antennas, trance_len)
            shifted_times_ (np.ndarray): times of the begining of signals from every source. shape (n_sources, n_antennas)
        outputs:
            list of summed signal. list of np.ndarray of shape : (4, len_out_trace)
    """
    if weights is None:
        weights = np.ones(shifted_times.shape)
    _tstep = signals[0,0,1] - signals[0,0,0] #ns
    trace_len = signals.shape[-1]
    n_sources, n_antennas = shifted_times.shape
    trace_duration = trace_len * _tstep
    output = np.zeros((n_sources, 4, trace_len))
    output[:,0,:] = np.arange(trace_len)[None,:] * _tstep + shifted_times[:,reference_antenna,None]
    for source_idx, source_times in enumerate(shifted_times):
        _sumef = np.zeros((3, trace_len))
        ref_time = source_times[reference_antenna]

        _start_idx = np.int_((source_times - ref_time)/_tstep)   #Start idx for each antenna
        _start_idx = np.clip(_start_idx, 0, trace_len-1)

        _end_idx = _start_idx+trace_len #Last idx for each antenna (length of traces)
        _end_idx = np.clip(_end_idx, 0, trace_len)

        for anti in range(n_antennas):
            _sumef[0, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[1,anti,:_end_idx[anti]-_start_idx[anti]]
            _sumef[1, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[2,anti,:_end_idx[anti]-_start_idx[anti]]
            _sumef[2, _start_idx[anti]:_end_idx[anti]] += weights[source_idx, anti] * signals[3,anti,:_end_idx[anti]-_start_idx[anti]]
        
        output[source_idx, 1:4,:] = _sumef
    return output


def spherical_phasing(xsrc, ysrc, zsrc, xant, yant, zant, timings):
    """
        Compute timing shifted for source position (xsrc, ysrc, zrsc) at antenna location (xant, yant, zant)
        inputs:
            xsrc (float or np.ndarray): source x position. Can be float for single source or ndarray of shape (n_sources) for parralel
            ysrc (float or np.ndarray): source y position. Can be float for single source or ndarray of shape (n_sources) for parralel
            zsrc (float or np.ndarray): source z position. Can be float for single source or ndarray of shape (n_sources) for parralel
            xant (np.ndarray): antenna x position. shape (n_antennas)
            yant (np.ndarray): antenna y position. shape (n_antennas)
            zant (np.ndarray): antenna z position. shape (n_antennas)
            timings (np.ndarray): initial timing of first time bin in simualtion shape: (n_antennas)
    """
    if isinstance(xsrc, (float, np.number)):
        xsrc = np.array([xsrc])
        ysrc = np.array([ysrc])
        zsrc = np.array([zsrc])
    assert xsrc.ndim == 1
    _ctdelays = np.sqrt((xant[None,:]-xsrc[:,None])**2+(yant[None,:]-ysrc[:,None])**2+(zant[None,:]-zsrc[:,None])**2) #shape (nsources, nantennas)
    

    ### This takes a long time. See if necessary to copy signals each time

    # _neff = np.array([GetZHSEffectiveRefractionIndex(xsrc, ysrc, zsrc,xant=xant[i],yant=yant[i],zant=zant[i],ns=325,kr=-0.1218,stepsize = 20000) for i in range(len(xant))])
    _neff_vect = ZHSEffectiveRefractionIndexvectTwice(np.array([xsrc, ysrc, zsrc]).T, np.array([xant, yant, zant]).T, ns=325, kr=-0.1218, stepsize = 20000) #shape (nsources, n_antennas)
    #6 times faster when vect


    new_times = timings[None,:] - _ctdelays/kc_ns*_neff_vect    #shape (n_sources, nantennas)

    return new_times

def spherical_beamforming(xsrc, ysrc, zsrc, xant, yant, zant, signals, reference_antenna=None):
    """
        Compute resulting traces from different sources locations, and antenna positions
        inputs:
            xsrc_ (float or np.ndarray): source x position. Can be float for single source or ndarray of shape (n_sources) for parralel
            ysrc_ (float or np.ndarray): source y position. Can be float for single source or ndarray of shape (n_sources) for parralel
            zsrc_ (float or np.ndarray): source z position. Can be float for single source or ndarray of shape (n_sources) for parralel
            xant_ (np.ndarray): antenna x position. shape (n_antennas)
            yant_ (np.ndarray): antenna y position. shape (n_antennas)
            zant_ (np.ndarray): antenna z position. shape (n_antennas)
            signals_ (np.ndarray): signal at for every antenna. shape (4, nants, trace_len)
        return:
            _shifted_times (np.ndarray): First time bin per event. shape (n_sources, n_antennas)
            _beamformed_signals (np.ndarray): summed signal with corresponding amplitudes. shape (n_sources, 4, output_trace_len)
    """
    _shifted_times = spherical_phasing(xsrc, ysrc, zsrc, xant, yant, zant, signals[0, :, 0])
    _beamformed_signals = signals_summing(signals, _shifted_times)
    # if reference_antenna is None:
    #     reference_antenna = np.argmax(signals[1:].max(axis=-1).sum(axis=0))
    # _beamformed_signals = signals_summing_ref(signals, _shifted_times, reference_antenna=reference_antenna)
    return _shifted_times, _beamformed_signals

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