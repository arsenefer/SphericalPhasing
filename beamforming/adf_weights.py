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
    return(omega_cr)

def ADF_parameters(theta, phi, delta_omega, Xants, Xsource, Bn=Bn):
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

    K = shower_direction_vector(theta, phi)
   
    asym_coeff = -0.003*np.rad2deg(theta)+0.220

    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bn)**2)

    l_ant = distance_source_antenna(Xants, Xsource)
    eta = eta_compute(theta, phi, Bvec, Xants, Xsource)
    omega = omega_compute(theta, phi, Xants, Xsource)
    omega_cr = [compute_Cerenkov_classic(Xsource)]*len(Xants)
    adf = 1/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
    adf *= 1. + asym*np.cos(eta) # 
    
    return eta, omega, omega_cr, l_ant, adf

def compute_adf_weights(xs, ys, zs, theta, phi, xant, yant, zant, delta_omega=0.2):
    """
    Compute ADF weights for given source and antenna positions.

    Parameters:
        xs, ys, zs (float): Source coordinates.  float (1)
        theta, phi (float): Source direction angles in radians. float (1)
        xant, yant, zant (ndarray): Antenna coordinates. shape (n_antennas,)
    Returns:
        ndarray: Computed ADF weights for each antenna. shape (n_antennas)
    """
    eta, omega, omega_cr, l_ant, adf = ADF_parameters(theta, phi, delta_omega=delta_omega,
                                                     Xants=np.stack((xant, yant, zant), axis=-1),
                                                     Xsource=np.array([xs, ys, zs]))
    weights = adf

    return weights
