import os
import subprocess
from itertools import product

energies = ["LE"]
treatment = ["N_F"]
method = ['both']

# Number of parallel workers per job (should match --cpus-per-task or be lower)

for energy, treat, meth in product(energies, treatment, method):
    if energy == "LE":
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    else:
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"

    interf_path = f"/sps/grand/aferrier/HERON_results/classical_bf/samples_samplingrandom_nwalkers200_nsteps600_stepsize1000_boundsflat_sphere_intensamplitude_temp5000"
    save_path = f"/sps/grand/aferrier/HERON_results/full_recons_{meth}"
    f_min, f_max = (50, 200) if "F" in treat else (0, 5000)
    noise_std = 13 if "N" in treat else 0

    args = [
        'sbatch', 'scripts/sequential_recons_job.sh',
        f'--path_to_library', f'{path_to_library}',
        f'--save_path', f'{save_path}',
        f'--interf_path', f'{interf_path}',
        f'--f_min', str(f_min),
        f'--f_max', str(f_max),
        f'--noise_std', str(noise_std),
        '--intens_method', 'amplitude',
        '--norm_weights', 'True',
        '--prior', str(True if meth in ["prior", "both"] else False),
        '--constant', str(True if meth in ["prior", "both"] else True),
    ]

    print('Submitting:', ' '.join(args))
    subprocess.run(args)
    