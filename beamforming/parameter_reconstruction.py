import numpy as np
import matplotlib.pyplot as plt

#### Reconstruction
def max_recons(all_walker_pos, signal_intesity):
    #find the best direction == highest signal amplitude
    best_rec = np.unravel_index(signal_intesity.argmax(), signal_intesity.shape)
    best_xsrc, best_ysrc, best_zsrc = all_walker_pos[best_rec[0], best_rec[1]]
    return best_xsrc, best_ysrc, best_zsrc, best_rec

def mean_recons(all_walker_pos, signal_intesity, weight=True):
    if weight:
        best_xsrc, best_ysrc, best_zsrc = np.average(all_walker_pos, axis=(0, 1), weights=np.repeat(signal_intesity[...,None], 3, axis=-1))
    else:
        best_xsrc, best_ysrc, best_zsrc = np.mean(all_walker_pos, axis=(0, 1))
    return best_xsrc, best_ysrc, best_zsrc, None



def compute_errors_source(best_xsrc, best_ysrc, best_zsrc, xmax_pos, tau_pos, shower_dir):
    rec_pos = np.array([best_xsrc, best_ysrc, best_zsrc])
    laterr = lateral_error(rec_pos, xmax_pos, shower_dir)
    longerr = longitudinal_error(np.array([best_xsrc, best_ysrc, best_zsrc]), xmax_pos, shower_dir)
    maxtaudist = np.linalg.norm(xmax_pos - tau_pos)
    relerr = longerr / maxtaudist
    return rec_pos, laterr, longerr, relerr

####### 

def make_proba_array(beamformed_intensity_array, T=None, threshold=0.):
    proba_array = beamformed_intensity_array - beamformed_intensity_array.min()
    T = T if T is not None else proba_array.std()
    proba_array = np.exp(proba_array / T)
    proba_array[beamformed_intensity_array < threshold] = 0.
    proba_array /= proba_array.sum()
    return proba_array


def _get_used_walkers(walker_positions, beamformed_intensity_array, n_walkers_to_consider=10, burn_in=0):
    """
    Get the walker positions and intensities to be used for direction reconstruction.
    If beamformed_intensity_array is provided, use the top n_walkers_to_consider walkers after burn-in.
    Otherwise, use all walkers after burn-in.
    """
    walker_pos_burnined = walker_positions[burn_in:]
    beam_formed_intensity_burnined = beamformed_intensity_array[burn_in:]
    best_walker_indexes = beam_formed_intensity_burnined.mean(axis=0).argsort()[-n_walkers_to_consider:]
    used_walker_positions = walker_pos_burnined[:, best_walker_indexes, :]
    used_intensities = beam_formed_intensity_burnined[:, best_walker_indexes]
    return used_walker_positions, used_intensities

def direction_from_cloud(positions, intensities):    
    Cov_matrix = np.cov(positions.reshape(-1, 3), aweights=intensities.flatten(), ddof=0, rowvar=False)
    eig_val, eig_vect = np.linalg.eig(Cov_matrix)
    k_rec = eig_vect[:,0]
    k_rec = k_rec if k_rec[1]>1 else -k_rec        
    k_rec /= np.linalg.norm(k_rec) 
    theta_rec, phi_rec = np.arccos(-k_rec[2]), np.arctan2(-k_rec[1], -k_rec[0])
    return k_rec, (theta_rec, phi_rec)

def direction_recons(walker_positions, beamformed_intensity_array, n_walkers_to_consider=10, sep_walkers=False, burn_in=0, weighted=True):
    """
    Reconstruct the direction from the walker positions.
    If beamformed_intensity_array is provided, use a weighted average of the top n_walkers_to_consider walkers.
    Otherwise, use a simple average of all walker positions.
    """
    used_walker_positions, used_intensities = _get_used_walkers(walker_positions, beamformed_intensity_array, n_walkers_to_consider, burn_in)
    used_walker_positions = used_walker_positions.reshape(-1, 3)
    used_intensities = used_intensities.reshape(-1,1)
    used_intensities = used_intensities[~np.isnan(used_walker_positions).any(axis=1)]
    used_walker_positions = used_walker_positions[~np.isnan(used_walker_positions).any(axis=1)]
    
    if weighted:
        weights = used_intensities
    else:
        weights = np.int_(used_intensities>0)

    if not sep_walkers:
        k_rec, (theta_rec, phi_rec) = direction_from_cloud(used_walker_positions, weights)
    else:
        k_recs, theta_recs, phi_recs, weights_dir = [], [], [], []
        for i in range(used_walker_positions.shape[1]):
            k_rec, (theta_rec, phi_rec) = direction_from_cloud(used_walker_positions[:,i], weights[:,i])
            k_recs.append(k_rec)
            theta_recs.append(theta_rec)
            phi_recs.append(phi_rec)
            weights_dir.append(weights[:,i].sum())
        k_rec = np.average(k_recs, axis=0, weights=weights_dir)
        k_rec /= np.linalg.norm(k_rec)
        theta_rec = np.average(theta_recs, weights=weights_dir)
        phi_rec = np.average(phi_recs, weights=weights_dir)
    return k_rec, (theta_rec, phi_rec)




def get_angular_distance(true_azim, rec_azim, true_zen, rec_zen):
    return np.arccos(np.cos(rec_zen*np.pi/180.)*np.cos(true_zen*np.pi/180.)+np.cos((rec_azim - true_azim)*np.pi/180.)*np.sin(true_zen*np.pi/180.)*np.sin(rec_zen*np.pi/180.))*180./np.pi

################################################################################
## plottig functions

def checkplots_sphericalrec(beamformed_maximum_map, x_start, x_stop, y_start, y_stop, z_start, z_stop, rec_xsrc, rec_ysrc, rec_zsrc, event_id, check_plots_dir):

    fx, (x1x, x2x, x3x) = plt.subplots(1, 3, figsize=(15, 8))
    x1x.set_xlabel(r"$\rm x\ (km)$") ; x1x.set_ylabel(r"$\rm y\ (m)$")
    x2x.set_xlabel(r"$\rm x\ (km)$") ;x2x.set_ylabel(r"$\rm z\ (m)$")
    x3x.set_xlabel(r"$\rm y\ (m)$") ; x3x.set_ylabel(r"$\rm z\ (m)$")

    xymap = np.sum(beamformed_maximum_map, axis=2)
    xymap /=np.amax(xymap)
    x1 = x1x.imshow(np.flipud(xymap.T), aspect='auto', extent=(x_start/1.e3, x_stop/1.e3, y_start, y_stop))
    cbarx1 = fx.colorbar(x1, ax=x1x)
    x1x.scatter(rec_xsrc/1.e3, rec_ysrc, color='red', label='recons')
    x1x.legend()

    xzmap = np.sum(beamformed_maximum_map, axis=1)
    xzmap /=np.amax(xzmap)
    x2 = x2x.imshow(np.flipud(xzmap.T), aspect='auto', extent=(x_start/1.e3, x_stop/1.e3, z_start, z_stop))
    cbarx2 = fx.colorbar(x2, ax=x2x)
    x2x.scatter(rec_xsrc/1.e3, rec_zsrc, color='red', label='recons')
    x2x.legend()

    yzmap = np.sum(beamformed_maximum_map, axis=0)
    yzmap /=np.amax(yzmap)
    x3 = x3x.imshow(np.flipud(yzmap.T), aspect='auto', extent=(y_start, y_stop, z_start, z_stop))
    cbarx3 = fx.colorbar(x3, ax=x3x)
    x3x.scatter(rec_ysrc, rec_zsrc, color='red', label='recons')
    x3x.legend()

    fx.savefig(check_plots_dir +"/rec_xyz_spherical_"+str(event_id)+".pdf", format='pdf')
    plt.close(fx)

    return 1

################################################################################
## Errors and statistics functions

def lateral_error(rec_pos, true_pos, shower_dir):
    return np.linalg.norm(np.cross(rec_pos -true_pos, shower_dir))

def longitudinal_error(rec_pos, true_pos, shower_dir):
    return np.dot(rec_pos -true_pos, shower_dir)
