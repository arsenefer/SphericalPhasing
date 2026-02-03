import numpy as np
from .utils import Bn


def shower_direction_vector(theta, phi):
    """
    Returns the shower direction vector K (orientation of the shower front).

    theta : zenith angle in radians
    phi   : azimuth angle in radians

    Returns:
        K : numpy array of shape (3,)
    """

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    K = np.array([
        -st * cp,
        -st * sp,
        -ct
    ])

    return K

def transformation_matrix(theta, phi, Bvec):
    """
    Builds the rotation matrix to the shower frame.

    theta : zenith angle (rad)
    phi   : azimuth angle (rad)
    Bvec  : global magnetic field vector (3,)
    """

    K = shower_direction_vector(theta, phi)

    KxB = np.cross(K, Bvec)
    KxB /= np.linalg.norm(KxB)

    KxKxB = np.cross(K, KxB)
    KxKxB /= np.linalg.norm(KxKxB)

    M = np.vstack((KxB, KxKxB, K))   # 3x3

    return M

def to_shower_frame(theta, phi, Bvec, Xant, Xsource):
    """
    Transforms global coordinates -> shower frame.

    Xant     : (3,) or (N,3) antenna positions
    Xsource  : (3,) core/Xmax position
    Returns:
        X_shower : coordinates in the shower frame
    """

    M = transformation_matrix(theta, phi, Bvec)
    dX = Xant - Xsource              # N x 3
    # projection onto shower frame
    X_shower = dX @ M.T              # N x 3 (=np.dot(mat,dX))
    return X_shower

def eta_compute(theta, phi, Bvec, Xants, Xsource):
    """
    Computes the angle eta (azimuth angle in the shower plane).

    theta  : shower zenith angle (rad)
    phi    : shower azimuth angle (rad)
    Bvec   : magnetic field vector (3,)
    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX_sp = to_shower_frame(theta, phi, Bvec, Xants, Xsource)
    if dX_sp.ndim == 1:
        return np.arctan2(dX_sp[1], dX_sp[0])
    else:
        return np.arctan2(dX_sp[:,1], dX_sp[:,0])

def distance_source_antenna(Xants,Xsource):
    """
    Computes the distance(s) between source and antenna(s).

    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX = Xants - Xsource 
    if dX.ndim == 1:
        return np.linalg.norm(dX)
    else:
        return np.linalg.norm(dX, axis=1)
    

def omega_compute(theta, phi, Xants, Xsource):
    """
    Computes the angle omega between shower direction and antenna vector(s).

    theta  : shower zenith angle (rad)
    phi    : shower azimuth angle (rad)
    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX = Xants - Xsource 
    l_ant = distance_source_antenna(Xants,Xsource)
    K = shower_direction_vector(theta, phi)
    if dX.ndim == 1:
        cos_omega = np.dot(K, dX) / l_ant   
    else:
        cos_omega = np.einsum('i,ni->n', K, dX) / l_ant 

    return np.arccos(cos_omega)  

def sin_geomag_angle(theta, phi, B=Bn):
    """
    Computes the sine of the geomagnetic angle (alpha) between the shower axis
    and the geomagnetic field.

    Parameters
    ----------
    theta : float or array-like
        Zenith angle(s) of the shower in radians.
    phi : float or array-like
        Azimuth angle(s) of the shower in radians.
    B : array-like, shape (3,), optional
        Geomagnetic field vector (default: cons.Bn).

    Returns
    -------
    sin_alpha : float or ndarray
        Sine of the geomagnetic angle
    """
    K = shower_direction_vector(theta, phi)
    sin_alpha = np.cross(K.T,B)
    sin_alpha = np.linalg.norm(sin_alpha)
    return sin_alpha
    