import os
import subprocess
from itertools import product

energies = ["LE"]
treatment = ["N_F"]
method = ['both']
jitter_std = [1, 2, 5] #in ns

# Number of parallel workers per job (should match --cpus-per-task or be lower)

for energy, treat, meth, jitter in product(energies, treatment, method, jitter_std):
    if energy == "LE":
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    else:
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"

    interf_path = f"/sps/grand/aferrier/HERON_results/classical_bf/samples_samplingrandom_nwalkers200_nsteps600_stepsize1000_boundsflat_sphere_intensamplitude_temp5000"
    save_path = f"/sps/grand/aferrier/HERON_results/global_recons_J{jitter}_{energy}"
    f_min, f_max = (50, 200) if "F" in treat else (0, 5000)
    noise_std = 13 if "N" in treat else 0

    args = [
        'sbatch', 'scripts/job_global_recons.sh',
        '--path_to_library', path_to_library,
        '--save_path', save_path,
        '--f_min', str(f_min),
        '--f_max', str(f_max),
        '--noise_std', str(noise_std),
        '--intens_method', 'amplitude',
        '--norm_weights', 'True',
        '--prior', str(True if meth in ["prior", "both"] else False),
        '--constant', str(True if meth in ["prior", "both"] else True),
        '--jitter_std', str(jitter_std)
    ]

    print('Submitting:', ' '.join(args))
    subprocess.run(args)
    