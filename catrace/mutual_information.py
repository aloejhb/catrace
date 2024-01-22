import os
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.feature_selection import mutual_info_regression

from sklearn.preprocessing import MinMaxScaler
from os.path import join as pjoin

from . import exp_collection as ecl

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MutualInfoConfig:
    pass

def compute_mi_experiment(dfx, dfy, parallelism=None):
    x = dfx.to_numpy() # OB
    y = dfy.to_numpy() # Dp

    # Initialize and fit the scaler
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    # Now compute the mutual information matrix
    mi_matrix = compute_mi_matrix_continuous(x_scaled, y_scaled, parallelism=parallelism)
    return mi_matrix


def compute_single_column(x, y_column):
    return mutual_info_regression(x, y_column, discrete_features=False)


def compute_mi_matrix_continuous(x, y, parallelism=None):
    """
    Compute the mutual information matrix between each feature in x and each feature in y.

    Parameters:
    x: numpy array of shape (n_samples, n_features_x)
    y: numpy array of shape (n_samples, n_features_y)
    parallelism: int, optional
        The number of worker processes to use for parallel computation.
        If None, use all available CPUs.

    Returns:
    mi_matrix: numpy array of shape (n_features_x, n_features_y)
        The mutual information matrix.
    """
    if parallelism is None:
        parallelism = cpu_count()

    n_features_x = x.shape[1]
    n_features_y = y.shape[1]
    mi_matrix = np.zeros((n_features_x, n_features_y))

    args_list = [(x, y[:, j]) for j in range(n_features_y)]

    with Pool(processes=parallelism) as pool:
        results = pool.starmap(compute_single_column, args_list)

    for j, mi_values in enumerate(results):
        mi_matrix[:, j] = mi_values

    return mi_matrix


def select_high_mi(db_dir, trace_dir, exp_name, mi_threshold):
    dfob = ecl.read_df(trace_dir, exp_name, 'OB', db_dir)
    dfdp = ecl.read_df(trace_dir, exp_name, 'Dp', db_dir)

    if trace_dir == 'dfovf':
        dfob.columns = dfob.columns.set_names('time')
        dfob = dfob.stack('time').unstack(['plane', 'neuron'])
        dfdp.columns = dfdp.columns.set_names('time')
        dfdp = dfdp.stack('time').unstack(['plane', 'neuron'])

    mi_file = pjoin(db_dir, 'mutual_information', f"mi_matrix_{exp_name}.npy")
    mi_matrix = np.load(mi_file)

    top_indices = np.argwhere(mi_matrix >= mi_threshold)

    ob_neurons = list(set(top_indices[:, 0]))
    dp_neurons = list(set(top_indices[:, 1]))
    print('OB #{} neurons'.format(len(ob_neurons)))
    print('Dp #{} neurons'.format(len(dp_neurons)))

    dfob_mi = dfob.iloc[:, ob_neurons]
    dfdp_mi = dfdp.iloc[:, dp_neurons]

    return dfob_mi, dfdp_mi, top_indices


def select_high_resp_low_mi(db_dir, trace_dir, exp_name, region, mi_thresh, resp_thresh):
    """
    Select neurons that have high response but low mutual information between OB and Dp
    """
    df = ecl.read_df(trace_dir, exp_name, region, db_dir)
    if trace_dir == 'dfovf':
        df.columns = df.columns.set_names('time')
        df = df.stack('time').unstack(['plane', 'neuron'])

    mi_file = pjoin(db_dir, 'mutual_information', f"mi_matrix_{exp_name}.npy")
    mi_matrix = np.load(mi_file)

    top_indices = np.argwhere(mi_matrix >= mi_threshold)

    if region == 'OB':
        region_idx = 0
    else:
        region_idx = 1

    high_mi_idx = list(set(top_indices[:, region_idx]))
    print('{} #{} neurons'.format(region, len(high_mi_idx)))

    df_select, high_resp_id = ptt.select_neuron(df, resp_thresh)
    high_resp_idx = np.where(high_resp_id)[0]

    high_resp_low_mi = np.setdiff1d(high_resp_idx, high_mi_idx)

    return high_resp_low_mi, df


def load_mi(mi_dir, exp_name):
    mi_file = os.path.join(mi_dir, f'{exp_name}.npy')
    mi_matrix = np.load(mi_file)
    return mi_matrix
