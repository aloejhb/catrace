import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations, product
from os.path import join as pjoin

from . import exp_collection as ecl
from . import process_time_trace as ptt


def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)


def compute_angles(df):
    results = []
    for index_a in df.index:
        point_a = df.loc[index_a].values
        for index_b, index_c in combinations(df.index, 2):
            if index_a != index_b and index_a != index_c:
                point_b = df.loc[index_b].values
                point_c = df.loc[index_c].values
                vector_ab = point_b - point_a
                vector_ac = point_c - point_a
                angle = angle_between_vectors(vector_ab, vector_ac)
            else:
                angle = np.nan

            results.append(((index_a, index_b, index_c), angle))

    return results


def compute_mahal_angles(trace_dir, pca_dir, exp_name, region, window, odor_list, eigenvec_num=0):
    dff = ecl.read_df(trace_dir, exp_name, region)

    # For each pair of odors
    odor_pairs = list(product(odor_list, odor_list))
    center_df = ptt.select_time_points(dff, window).groupby('odor').mean()
    angles = {}
    for src_odor, ref_odor in odor_pairs:
        # Get the first eigenvector of the reference odor
        odor_pca_dir = re.sub(r'_odors_([a-zA-Z]+)_', f'_odors_{ref_odor}_', pca_dir)
        model_file = pjoin(odor_pca_dir, 'models', f'{exp_name}_{region}.pkl')
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        ref_egvec = model.components_[eigenvec_num]

        # Get the vector from the source odor to reference odor
        src_center = center_df.loc[src_odor]
        ref_center = center_df.loc[ref_odor]
        diff_vec = (src_center - ref_center).to_numpy()

        # Compute the angle
        angle = angle_between_vectors(ref_egvec, diff_vec)
        angles[(src_odor, ref_odor)] = angle
    angles_mat = dict_to_matrix(angles, odor_list)
    return angles_mat


def dict_to_matrix(data, odor_list):
    """
    Convert a dictionary with tuple keys to a matrix represented as a pandas DataFrame.

    Parameters:
    - data: Dictionary with keys as tuples (row_label, col_label) and values as integers.

    Returns:
    - A pandas DataFrame representing the matrix.
    """
    # Convert dictionary to DataFrame with 'Value' as the column for dictionary values
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value']).reset_index()
    # Split the 'index' column (containing tuples) into two separate columns for 'Row' and 'Column'
    df[['Row', 'Column']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    df.drop(columns=['index'], inplace=True)

    # Pivot the DataFrame to create the matrix, specifying 'Row' as the index, 'Column' as the columns,
    # and 'Value' as the values to populate the matrix
    matrix_df = df.pivot(index='Row', columns='Column', values='Value')

    # Fill NaN values with 0 to ensure all cells have a value and convert to integers
    matrix_df = matrix_df.fillna(0).astype(int)

    # Sort rows
    matrix_df.index = pd.Categorical(matrix_df.index,
                                     categories=odor_list,
                                     ordered=True)
    matrix_df = matrix_df.sort_index()

    # Sort columns
    matrix_df = matrix_df[odor_list]
    return matrix_df


def plot_matrix(matrix, ax=None, vmin=None, vmax=None):
    ax = sns.heatmap(matrix, ax=ax, cmap="RdBu_r", cbar=False,
                     vmin=vmin, vmax=vmax,
                     xticklabels=matrix.columns, yticklabels=matrix.index)
    return ax
