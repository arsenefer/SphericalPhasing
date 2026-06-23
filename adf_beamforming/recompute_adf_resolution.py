import os
import re
import argparse
import numpy as np
import pandas as pd
from math import pi

import beamforming_module.read_sims_module as rm
from beamforming_module.adf_weights import ADF_parameters, prior_adf

R2D = 180.0 / np.pi


def find_reconstruction_csv(maps_dir):
    for fn in os.listdir(maps_dir):
        if fn.endswith('_reconstruction_results.csv'):
            return os.path.join(maps_dir, fn)
    return None


def parse_event_id(fname):
    m = re.search(r"_(\d+)_intensity_map\.npy$", fname)
    if m:
        return int(m.group(1))
    return None


def main(maps_dir, library_path, delta_omega, prior_scale, norm_weights, constant, out_csv):
    if not os.path.isdir(maps_dir):
        raise FileNotFoundError(maps_dir)

    csv_path = find_reconstruction_csv(maps_dir)
    if csv_path is None:
        raise FileNotFoundError('reconstruction csv not found in ' + maps_dir)

    df_old = pd.read_csv(csv_path)

    # load library to get antenna positions
    tau_events, antennas, efields, sttimes = rm.read_library(library_path)
    antenna_pos = np.copy(antennas['position'])
    antenna_alt = np.array(antennas['coordinates'])[:, 2]
    antenna_pos[:, 2] += antenna_alt[0]
    xant, yant, zant = antenna_pos.T

    results = []

    files = sorted([f for f in os.listdir(maps_dir) if f.endswith('_intensity_map.npy')])
    for fn in files:
        event_id = parse_event_id(fn)
        if event_id is None:
            continue
        map_path = os.path.join(maps_dir, fn)
        try:
            intensity_map = np.load(map_path)
        except Exception:
            print('skipping', map_path)
            continue

        row = df_old.loc[df_old['event_id'] == event_id]
        if row.empty:
            print('no metadata for event', event_id)
            continue
        row = row.iloc[0]

        true_theta = float(row['true_theta'])
        true_phi = float(row['true_phi'])
        xmax_pos = np.array([row['xmax_pos_x'], row['xmax_pos_y'], row['xmax_pos_z']])
        tau_pos = np.array([row['tau_pos_x'], row['tau_pos_y'], row['tau_pos_z']])

        n_theta, n_phi = intensity_map.shape
        range_theta = np.linspace(-4, 4, n_theta) * pi / 180.0 + true_theta
        range_phi = np.linspace(-4, 4, n_phi) * pi / 180.0 + true_phi

        # build prior map
        prior_map = np.zeros_like(intensity_map)
        for i, th in enumerate(range_theta):
            for j, ph in enumerate(range_phi):
                eta, omega, omega_cr, l_ant, weights = ADF_parameters(
                    th, ph, delta_omega=delta_omega,
                    Xants=np.stack((xant, yant, zant), axis=-1),
                    Xsource=xmax_pos,
                    norm_weights=norm_weights, constant=constant)
                prior_map[i, j] = prior_adf(weights, omegas=omega, omega_cr=omega_cr)

        # add scaled prior
        modified_map = intensity_map + prior_scale * prior_map

        # find maximum and compute metrics
        max_idx = np.unravel_index(np.argmax(modified_map), modified_map.shape)
        theta_max = range_theta[max_idx[0]]
        phi_max = range_phi[max_idx[1]]

        # reconstruct layout and shower_dir
        k_layout = None
        try:
            # reuse layout_phi/layout_theta from old row if present
            layout_phi = float(row.get('layout_phi', np.nan))
            layout_theta = float(row.get('layout_theta', np.nan))
        except Exception:
            layout_phi = np.nan
            layout_theta = np.nan

        shower_dir = (xmax_pos - tau_pos)
        shower_dir = shower_dir / np.linalg.norm(shower_dir)

        k_rec = -np.array([np.sin(theta_max) * np.cos(phi_max),
                           np.sin(theta_max) * np.sin(phi_max),
                           np.cos(theta_max)])

        angular_error_deg = np.arccos(np.clip((shower_dir * k_rec).sum(), -1.0, 1.0)) * R2D

        out_row = {
            'event_id': int(event_id),
            'true_phi': float(true_phi),
            'true_theta': float(true_theta),
            'layout_phi': layout_phi,
            'layout_theta': layout_theta,
            'max_intensity': float(np.max(modified_map)),
            'reco_phi': float(phi_max),
            'reco_theta': float(theta_max),
            'theta_error_deg': float((theta_max - true_theta) * R2D),
            'phi_error_deg': float((phi_max - true_phi) * R2D),
            'angular_error_deg': float(angular_error_deg),
            'xmax_pos_x': float(xmax_pos[0]),
            'xmax_pos_y': float(xmax_pos[1]),
            'xmax_pos_z': float(xmax_pos[2]),
            'tau_pos_x': float(tau_pos[0]),
            'tau_pos_y': float(tau_pos[1]),
            'tau_pos_z': float(tau_pos[2]),
        }

        # copy over available metrics from old dataframe if present
        for key in ['max_amp', 'ave_amp', 'nb_above5sigma', 'nb_above3sigma', 'nb_above1sigma', 'energy', 'en_nu']:
            if key in row.index:
                out_row[key] = row[key]

        results.append(out_row)

    df_new = pd.DataFrame(results)
    df_new = df_new.sort_values('event_id')
    df_new.to_csv(out_csv, index=False)
    print('Wrote', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps_dir', required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--delta_omega', type=float, default=1.2)
    parser.add_argument('--prior_scale', type=float, default=100.0)
    parser.add_argument('--norm_weights', action='store_true', default=True)
    parser.add_argument('--no-norm_weights', dest='norm_weights', action='store_false')
    parser.add_argument('--constant', action='store_true', default=True)
    parser.add_argument('--no-constant', dest='constant', action='store_false')
    parser.add_argument('--out_csv', default=None)
    args = parser.parse_args()

    out_csv = args.out_csv or os.path.join(args.maps_dir, 'reconstruction_results_montain.csv')
    main(args.maps_dir, args.library, args.delta_omega, args.prior_scale, args.norm_weights, args.constant, out_csv)
