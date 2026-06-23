import numpy as np
def xyz_parameter_bounds(tau_pos_, phi, theta):

    nphi, ntheta, nd = 100, 100, 100
    dphi, dtheta = 0.2*np.pi/180., 0.2*np.pi/180.

    phi_range, theta_range, d = np.linspace(phi-dphi, phi+dphi, nphi), np.linspace(theta-dtheta, theta+dtheta, ntheta), np.linspace(0, 10.e3, nd)

    xmat = tau_pos_[0]+np.cos(phi_range)*np.sin(theta_range)*d[:,None]
    ymat = tau_pos_[1]+np.sin(phi_range)*np.sin(theta_range)*d[:,None]
    zmat = tau_pos_[2]+np.cos(theta_range)*d[:,None]

    x_bounds = np.array([xmat.min(), xmat.max()])
    y_bounds = np.array([ymat.min(), ymat.max()])
    z_bounds = np.array([zmat.min(), zmat.max()])

    return x_bounds, y_bounds, z_bounds

def cubic_bound_function(x_bounds, y_bounds, z_bounds):
    def cubic_bound(x,y,z):
        in_x = (x_bounds[0]<=x) & (x<=x_bounds[1])
        in_y = (y_bounds[0]<=y) & (y<=y_bounds[1])
        in_z = (z_bounds[0]<=z) & (z<=z_bounds[1])
        return in_x & in_y & in_z
    return cubic_bound

def cubic_init(n_walker, x_bounds, y_bounds, z_bounds):
    X_0 = np.random.rand(n_walker, 3)
    X_0[:,0] = X_0[:,0]*(x_bounds[1]-x_bounds[0]) + x_bounds[0]
    X_0[:,1] = X_0[:,1]*(y_bounds[1]-y_bounds[0]) + y_bounds[0]
    X_0[:,2] = X_0[:,2]*(z_bounds[1]-z_bounds[0]) + z_bounds[0]
    return X_0

def sphere_bound_function(center_pos, radius=10e3):
    def sphere_bound(x, y, z):
        return (x - center_pos[0])**2 + (y - center_pos[1])**2 + (z - center_pos[2])**2 <= radius**2
    return sphere_bound

def sphere_init(n_walker, center_pos, radius=10e3):
    X_0 = np.random.randn(n_walker, 3)
    X_0 = X_0 / np.linalg.norm(X_0, axis=1, keepdims=True)
    r = np.cbrt(np.random.rand(n_walker))*radius
    X_0 = X_0 * r[:,None]
    X_0 += center_pos[None,:]
    return X_0

def flat_sphere_bound_function(center_pos, radius=10e3, thickness=500.):
    def flat_sphere_bound(x, y, z):
        r2 = ((x - center_pos[0])/radius)**2 + ((y - center_pos[1])/radius)**2 + ((z - center_pos[2])/thickness)**2
        return r2 <= 1.
    return flat_sphere_bound

def flat_sphere_init(n_walker, center_pos, radius=10e3, thickness=500.):
    X_0 = np.random.randn(n_walker, 3)
    X_0 = X_0 / np.linalg.norm(X_0, axis=1, keepdims=True)
    r = np.cbrt(np.random.rand(n_walker))
    X_0 = X_0 * r[:,None]
    X_0[:,0] *= radius
    X_0[:,1] *= radius
    X_0[:,2] *= thickness
    X_0 += center_pos[None,:]
    return X_0

################################################################################
##regular grid
def create_regular_grid(n_walkers, n_steps, Xrange, Yrange, Zrange, in_bound_function):
    #Create a cubic grid of points within the given ranges
    N_point_z = n_walkers
    N_point_xy = int(np.ceil(np.sqrt(n_steps)))

    x_vals = np.linspace(Xrange[0], Xrange[1], N_point_xy)
    y_vals = np.linspace(Yrange[0], Yrange[1], N_point_xy)
    z_vals = np.linspace(Zrange[0], Zrange[1], N_point_z)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    grid_points = np.concatenate([X[..., None], Y[..., None], Z[..., None]], axis=-1)
    grid_points = grid_points.reshape(N_point_xy**2, N_point_z, 3)
    grid_points[~in_bound_function(grid_points[..., 0], grid_points[...,1], grid_points[...,2])] = np.nan
    return grid_points

def sample_3planes(n_walkers, n_steps, center_pos, radius, shower_dir):
    # Create three planes ortogonal to the shower direction
    # And placed at center_pos +/- radius along the shower direction
    # On each plane, regular create a grid of points within a square of side 2*radius
    # The number of points on each plane is n_walkers * n_step // 3
    n_planes = 3
    n_points_per_plane = n_walkers * n_steps // n_planes
    n_points_per_ax = int(np.ceil(np.sqrt(n_points_per_plane)))
    side_length = 2 * radius
# n_points_per_ax = 50
# side_length = 2 * 50
# shower_dir = np.array([1, 0, 0])
    projected_X, projected_Y = np.mgrid[-side_length/2:side_length/2:n_points_per_ax*1j, -side_length/2:side_length/2:n_points_per_ax*1j]
    projected_points = np.stack([projected_X, projected_Y, np.zeros_like(projected_Y)], axis=-1)
    shower_dir = shower_dir / np.linalg.norm(shower_dir)
    if np.allclose(shower_dir, np.array([0, 0, 1])):
        ortho1 = np.array([1, 0, 0])
    else:
        ortho1 = np.cross(shower_dir, np.array([0, 0, 1]))
        ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(shower_dir, ortho1)
    ortho2 /= np.linalg.norm(ortho2)
    rotation_matrix = np.stack([ortho1, ortho2, shower_dir], axis=1)
    rotated_points = (rotation_matrix @ projected_points.reshape(3, -1)).reshape(n_points_per_ax**2, 3)
    plane_offsets = (center_pos + np.array([-radius, 0, radius]))[:, None] * shower_dir[None, :]

