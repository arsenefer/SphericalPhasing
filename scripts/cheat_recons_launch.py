import os
from itertools import product
energies = ["LE", "HE"]
treatment = ["N_F", "none"]
method = ['prior', 'constant', 'both']
for energy, treat, meth in product(energies, treatment, method):
    if energy == "LE":
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    else:
        path_to_library = "/sps/grand/vdecoene/HERON/NuSimLibrary/TauLibrary_250Events_Eshower_1e8-1e10GeV_XYZCoordinates_CorrXmax.npz"
    
    save_path = f"/sps/grand/aferrier/HERON_results/ADF_Fixed_'norm'{'_'+treat if treat != 'none' else ''}_{meth}"
    f_min, f_max = (50, 200) if "F" in treat else (0, 5000)
    noise_std = 13 if "N" in treat else 0
    
    os.system(
        f"""sbatch scripts/cheat_recons_job.sh \
            '--path_to_library {path_to_library} \
            --save_path {save_path} \
            --f_min {f_min} \
            --f_max {f_max} \
            --noise_std {noise_std} \
            --intens_method amplitude \
            --norm_weights True \
            --prior {True if meth in ["prior", "both"] else False} \
            --constant {True if meth in ["prior", "both"] else True}'""")
    