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
