import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# import beamforming.sequential.beamforming_module as bm
import beamforming_module.read_sims_module as rm
import beamforming_module.signal_module as sm
import beamforming_module.sampling_module_para as sp
import beamforming_module.parameter_reconstruction as rec

from beamforming_module.sampler_init import (cubic_bound_function, 
                                      cubic_init, 
                                      sphere_bound_function, 
                                      sphere_init, 
                                      xyz_parameter_bounds, 
                                      flat_sphere_bound_function, 
                                      flat_sphere_init,
                                      create_regular_grid)


PATH_TO_LIBRARY = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"

tau_events, antennas, efields, sttimes = rm.read_library(PATH_TO_LIBRARY)




# PATH_TO_LIBRARY = "/volatile/home/af274537/Documents/DATA/HERON/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates.npz"
# SAVE_PATH = './results'

# SAMPLING = "random"                                 # Can be "cubic_grid", "parallel_tempering", "metrop_hasting", "random"
# N_WALKERS = 200                                          # Set to an integer for "parallel_tempering" or "metrop_hasting" else None
# N_STEPS = 400                                           # Number of sampling steps should be > 1 if SAMPLING != "cubic_grid" else None
# CELL_SIZE = None                                        # Set to an integer for "cubic_grid" else None
# INTENS_METHOD = "amplitude"                                 # can be "amplitude" or "power"
# TEMP = 5000.                                            # Only for "metrop_hasting"
# F_MIN, F_MAX, NOISE_STD, JITTER_STD = 0, 5000, 0, 0     # Signal processing parameters
# STEP_SIZE = 1000.                                        # Step size for "parallel_tempering" and "metrop_hasting"

# BOUNDS = "flat_sphere"                                  # can be "cubic", "sphere", "flat_sphere"
# R = 10.e3                                               # Only for "sphere" and "flat_sphere"  
# THICKNESS = 4e3                                        # Only for "flat_sphere"
# XMAX_METHOD = 'max'                                     # can be 'max' or 'avg' or 'mode'
# N_BEST_WALKERS = 0                                     # Number of best walkers to consider for Xmax and direction estimation
# SEP_WALKERS = False                                     # Whether to separate walkers for direction estimation
# BURN_IN = 0  