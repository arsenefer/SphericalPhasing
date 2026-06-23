## compute_recons.py — Config fields

This file documents the fields of the `Config` dataclass defined in `compute_recons.py`.
Each entry lists the Python type, default value (as in the source), units when applicable, permitted options, and a short description of how the parameter is used.

---

### Config entries

- `path_to_library` (str)
	- Default: `/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates.npz`
	- Description: Path to the NPZ event/library file used by `read_sims_module.read_library`. This file contains the simulated events, antenna definitions, electric fields and timing data used for reconstructions.

- `save_path` (str)
	- Default: `./results`
	- Description: Directory where output (sample `.npz` files and `recons_base.csv`) will be saved. The script will create subdirectories under this path according to sampling parameters.

- `sampling` (str)
	- Default: `'random'`
	- Options: `'random'`, `'cubic_grid'`, `'metrop_hasting'`, `'parallel_tempering'`
	- Description: Sampling strategy used to generate walker positions and run beamforming/sampling. Controls which sampling routines in `sampling_module_para` are used.

- `n_walkers` (int)
	- Default: `200`
	- Description: Number of walkers (samples per step) used in MCMC / grid / random sampling.

- `n_steps` (int)
	- Default: `400`
	- Description: Number of sampling steps (time dimension). For grid sampling this is used as the number of grid points in one axis (script-specific behavior).

- `step_size` (float)
	- Default: `1000.`
	- Units: meters
	- Description: Typical step size scaling for proposing moves in sampling (per-axis scale is multiplied by this). Used to define `step_vect` in the main script.

- `f_min` (float)
	- Default: `0`
	- Units: Hz
	- Description: Minimum frequency passed to `read_trace` (signal processing / band limits).

- `f_max` (float)
	- Default: `5000`
	- Units: Hz
	- Description: Maximum frequency passed to `read_trace` (signal processing / band limits).

- `noise_std` (float)
	- Default: `0`
	- Units: same units as signal amplitude
	- Description: Standard deviation of additive noise to be applied when reading traces (for simulating measurement noise). If 0, no extra noise is added.

- `jitter_std` (float)
	- Default: `0`
	- Units: seconds (time jitter)
	- Description: Standard deviation of time jitter applied to simulated traces. Helps test robustness against timing errors.

- `bounds` (str)
	- Default: `'flat_sphere'`
	- Options: `'cubic'`, `'sphere'`, `'flat_sphere'`
	- Description: Defines the geometric parameter bounds used to initialize walkers:
		- `cubic`: use axis-aligned x/y/z ranges (calls `cubic_init`, `xyz_parameter_bounds`)
		- `sphere`: initialize on a sphere around `xmax_pos` with radius `r` (calls `sphere_init`)
		- `flat_sphere`: initialize on a spherical shell with limited thickness (calls `flat_sphere_init`)

- `r` (float)
	- Default: `10.e3` (10,000.0)
	- Units: meters
	- Description: Radius parameter used for spherical/flat spherical initializations (`sphere` and `flat_sphere` modes). Represents distance (m) from `xmax_pos` where walkers are placed.

- `thickness` (float)
	- Default: `4e3` (4,000.0)
	- Units: meters
	- Description: Thickness of the flat spherical shell when `bounds='flat_sphere'`. Controls allowed z-deviation from the shell.

- `intens_method` (str)
	- Default: `'amplitude'`
	- Likely options: `'amplitude'`, `'fluence'` (script references intensity as amplitude/fluence; check `beamforming` modules for exact supported strings)
	- Description: Method used to compute beamformed intensity from signals (affects likelihood or score used by sampling).

- `temp` (float)
	- Default: `5000.`
	- Description: Temperature parameter used by sampling routines (e.g., parallel tempering / MCMC acceptance). Higher values flatten the probability surface.

- `x_max_method` (str)
	- Default: `'max'`
	- Description: Controls method used to find the maximum in beamformed maps (left in config for compatibility with code paths that may select different methods).

- `n_best_walkers` (int)
	- Default: `0`
	- Description: Number of best walkers to consider when computing reconstructed direction (used by `parameter_reconstruction.direction_recons`). If 0 the function's default behavior is used.

- `sep_walkers` (bool)
	- Default: `False`
	- Description: When `True`, separate walker groups can be considered independently in direction reconstruction. Used by `direction_recons`.

- `burn_in` (int)
	- Default: `0`
	- Description: Number of initial steps to discard when plotting or analyzing walker traces (useful to remove transient initialization behavior).

---

Notes and usage
- Units: positions, `r`, `thickness`, and `step_size` are in meters. `f_min`/`f_max` are in Hz, `noise_std` is in signal amplitude units, and `jitter_std` is in seconds.
- Randomness: the main script seeds NumPy with the event index (`np.random.seed(i)`) for reproducibility across events in a single run.
- Output layout: the script saves per-event `.npz` files under `save_path` in a directory built from sampling parameters (e.g., `samples_samplingrandom_nwalkers200_...`). A `recons_base.csv` summary is also written into that same directory.

Quick run
- Edit any `Config` defaults in `compute_recons.py` or instantiate `Config(...)` with desired overrides in the `if __name__ == "__main__"` section, then run:

- `python compute_recons.py`

If you modify parameter names or add fields, update this README accordingly.

---

End of file.

