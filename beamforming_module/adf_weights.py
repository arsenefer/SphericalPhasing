import numpy as np
from .geom import shower_direction_vector, distance_source_antenna, eta_compute, omega_compute
from .utils import Bvec, Bn



def compute_Cerenkov_classic(Xsource, ns=325, kr=-0.1218):

    """
    Compute Cherenkov angle by minimizing the time delay between light rays from shower points and the observer.

    Inputs:
    - Xsource : np.array, shape (3,) -> position of Xsource 

    Returns:
    - omega_cr : float -> Cherenkov angle in radians
    """

    h0 = Xsource[2]/1e3
    rh0 = ns*np.exp(kr*h0) #refractivity at emission
    n_atm=1.E0+1.E-6*rh0

    omega_cr = np.arccos(1./n_atm)
    return omega_cr

def ADF_parameters(theta, phi, delta_omega, Xants, Xsource, Bn=Bn, norm_weights=False, constant=True):
    """
    Compute all geometric parameters for the ADF function.
    
    Inputs:
        theta, phi          : shower direction angles (rad) (from ADF_recons, best fit values)
        delta_omega         : ADF shape parameter (output from ADF_recons, best fit values)
        Xants  : (N,3) positions of antennas
        Xsource   : (3,) position of Xsource (from SWF)
        Bvec   : (3,) magnetic field
        groundAltitude : altitude of ground
    
    Returns:
        eta       : (N,) azimuthal angle in shower plane
        omega     : (N,) angle wrt shower axis
        omega_cr  : (N,) Cherenkov angle for each antenna
        l_ant     : (N,) distance from Xsource to antenna
        adf       : (N,) ADF amplitude for each antenna
    """


    l_ant = distance_source_antenna(Xants, Xsource)
    eta = eta_compute(theta, phi, Bn, Xants, Xsource)
    omega = omega_compute(theta, phi, Xants, Xsource)
    omega_cr = np.ones(len(Xants))*compute_Cerenkov_classic(Xsource)
    adf = 1/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)

    K = shower_direction_vector(theta, phi)
    asym_coeff = -0.48
    sin_alpha = np.sqrt(1. - np.dot(K,Bn)**2)
    asym = asym_coeff/sin_alpha
    adf *= 1. + asym*np.cos(eta) # 

    if constant:
        adf += 1/l_ant / 100
    if norm_weights:
        adf /= np.linalg.norm(adf)
    
    return eta, omega, omega_cr, l_ant, adf


def prior_adf(weights, omegas, omega_cr, alpha=1000):
    """
    Compute the ADF prior based on the difference between the observed angles and the Cherenkov angle.

    Parameters:
        weights (ndarray): ADF weights for each antenna. shape (n_antennas,)
        omegas (ndarray): Observed angles for each antenna. shape (n_antennas,)
        omega_cr (ndarray): Cherenkov angles for each antenna. shape (n_antennas,)
    Returns:
        float: Computed ADF prior value.
    """
    diff = np.abs(omegas - omega_cr)
    # prior_ant = np.exp(-alpha * diff)
    prior_ant = -alpha * diff**2
    return np.sum(weights * prior_ant)