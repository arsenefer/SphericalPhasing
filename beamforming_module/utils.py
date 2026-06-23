##custom display
import time
import sys
import numpy as np
import scipy.signal as ss

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def get_peaks_hilbert(signal):

    hilbert_env = np.linalg.norm(ss.hilbert(signal[1:,:,:]), axis=0)
    sel = np.argmax(hilbert_env, axis=1)[:, np.newaxis]
    peaka=np.take_along_axis(hilbert_env, sel, axis=1)
    peakt=np.take_along_axis(signal[0,:,:], sel, axis=1)

    return peakt.reshape(len(signal[0,:,:])), peaka.reshape(len(signal[0,:,:]))

def sph2cart(theta:np.ndarray, phi:np.ndarray, r=1):
    """
    Convert spherical coordinate to cartesian coordinate
    """
    if isinstance(theta, (np.floating, float, np.ndarray)):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.stack((x, y, z), axis=-1)
    elif type(theta) is torch.Tensor:
        x = r*torch.sin(theta)*torch.cos(phi)
        y = r*torch.sin(theta)*torch.sin(phi)
        z = r*torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)
    else:
        raise TypeError(f"Input must be a numpy array or a torch tensor, not {type(theta)}.")

def cart2sph(x:np.ndarray, y:np.ndarray, z:np.ndarray):
    """
    Convert cartesian coordinate to spherical coordinate
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return theta, phi
########Constants
Bvec =  [-20.16850382,  10.52718602, 0.37808577]
Bn = Bvec/np.linalg.norm(Bvec)

Bn[2] = -np.linalg.norm(Bn[:2])*np.tan(np.radians(35.06))
Bn = Bn/np.linalg.norm(Bn)