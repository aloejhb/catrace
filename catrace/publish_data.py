import os
import numpy as np
import pandas as pd
import h5py


from .run.run_pattern_similarity import read_mats_from_dir as read_pattern_similarity_mats_from_dir
from .run.run_distance import read_mats_from_dir, average_over_repeats


def save_distance_results_as_h5(metric_dir, metric_name, exp_list, num_repeats, output_dir):
    all_simdf = read_mats_from_dir(
        metric_dir, exp_list, num_repeats
    )
    avg_simdf = average_over_repeats(all_simdf)
    num_fish = len(exp_list)

    for idx, (exp_name, condition) in enumerate(exp_list):
        fish_mat = avg_simdf.xs(exp_name, level='fish_id')
        if idx == 0:
            simmat = np.zeros((num_fish, fish_mat.shape[0], fish_mat.shape[1]))
        simmat[idx, :, :] = fish_mat.values

    axis0 = exp_list
    axis1 = avg_simdf.index.get_level_values(-1).unique().tolist()
    axis2 = avg_simdf.columns.get_level_values(0).unique().tolist()
    levels_axis0 = ['fish_id', 'condition']
    levels_axis1 = [avg_simdf.index.names[-1]]
    levels_axis2 = [avg_simdf.columns.names[0]]

    # Put simmat, axis_names, axis_keys into an hdf5 dataset
    os.makedirs(output_dir, exist_ok=True)
    output_h5file = os.path.join(output_dir, f'{metric_name}_averaged_over_repeats.h5')
    with h5py.File(output_h5file, 'w') as h5f:
        h5f.create_dataset('simmat', data=simmat)
        h5f.create_dataset('axis0', data=np.array(axis0, dtype='S'))
        h5f.create_dataset('axis1', data=np.array(axis1, dtype='S'))
        h5f.create_dataset('axis2', data=np.array(axis2, dtype='S'))
        h5f.create_dataset('levels_axis0', data=np.array(levels_axis0, dtype='S'))
        h5f.create_dataset('levels_axis1', data=np.array(levels_axis1, dtype='S'))
        h5f.create_dataset('levels_axis2', data=np.array(levels_axis2, dtype='S'))


def read_distance_results_from_h5(input_path):
    """
    Read distance results from an HDF5 file with:
        simmat            (matrix)
        axis0             (row index levels)
        axis1             (column index levels)
        axis2             (third dimension index levels)
        levels_axis0      (names of row index levels)
        levels_axis1      (names of column index levels)
        levels_axis2      (names of third dimension index levels)
    """
    with h5py.File(input_path, "r") as h5f:
        simmat = h5f["simmat"][:]
        axis0 = h5f["axis0"][:].astype(str).tolist()
        axis1 = h5f["axis1"][:].astype(str).tolist()
        axis2 = h5f["axis2"][:].astype(str).tolist()
        levels_axis0 = h5f["levels_axis0"][:].astype(str).tolist()
        levels_axis1 = h5f["levels_axis1"][:].astype(str).tolist()
        levels_axis2 = h5f["levels_axis2"][:].astype(str).tolist()

    # Reconstruct for each fish
    simdfs = []
    for i in range(simmat.shape[0]):
        row_index = pd.Index(axis1, name=levels_axis1[0])
        col_index = pd.Index(axis2, name=levels_axis2[0])
        # Convert to DataFrame
        simdf = pd.DataFrame(
            simmat[i], index=row_index, columns=col_index
        )
        simdfs.append(simdf)
    # Concatenate simdfs
    axis0_tuples = [tuple(axis0[i]) for i in range(len(axis0))]
    all_simdf = pd.concat(simdfs, keys=axis0_tuples, names=levels_axis0)
    return all_simdf


def save_similarity_results_as_h5(metric_dir, metric_name, exp_list, output_dir):
    simdf_list, _ = read_pattern_similarity_mats_from_dir(
        metric_dir, exp_list
    )

    num_fish = len(exp_list)

    first_simdf = simdf_list[0]
    simmat = np.zeros((num_fish, first_simdf.shape[0], first_simdf.shape[1]))

    for idx, (exp_name, condition) in enumerate(exp_list):
        simmat[idx, :, :] = simdf_list[idx].values

    levels_axis0 = ['fish_id', 'condition']
    levels_axis1 = ['odor']
    levels_axis2 = ['odor']

    axis0 = exp_list
    axis1 = first_simdf.index.get_level_values('odor')
    axis2 = first_simdf.columns.get_level_values('odor')

    # Put simmat, axis_names, axis_keys into an hdf5 dataset
    output_h5file = os.path.join(output_dir, f'{metric_name}.h5')
    with h5py.File(output_h5file, 'w') as h5f:
        h5f.create_dataset('simmat', data=simmat)
        h5f.create_dataset('axis0', data=np.array(axis0, dtype='S'))
        h5f.create_dataset('axis1', data=np.array(axis1, dtype='S'))
        h5f.create_dataset('axis2', data=np.array(axis2, dtype='S'))
        h5f.create_dataset('levels_axis0', data=np.array(levels_axis0, dtype='S'))
        h5f.create_dataset('levels_axis1', data=np.array(levels_axis1, dtype='S'))
        h5f.create_dataset('levels_axis2', data=np.array(levels_axis2, dtype='S'))


def save_dataframe_as_h5(df, output_path):
    """
    Save any DataFrame into an HDF5 file with:
        data              (matrix)
        axis0             (row index levels)
        axis1             (column index levels)
        levels_axis0      (names of row index levels)
        levels_axis1      (names of column index levels)
    """

    if isinstance(df.index, pd.MultiIndex):
        row_levels = df.index.names
        row_values = np.column_stack([
            df.index.get_level_values(i).astype(str).values
            for i in range(df.index.nlevels)
        ])
    else:
        row_levels = [df.index.name if df.index.name is not None else "index"]
        row_values = df.index.astype(str).values.reshape(-1, 1)

    if isinstance(df.columns, pd.MultiIndex):
        col_levels = df.columns.names
        col_values = np.column_stack([
            df.columns.get_level_values(i).astype(str).values
            for i in range(df.columns.nlevels)
        ])
    else:
        col_levels = [df.columns.name if df.columns.name is not None else "columns"]
        col_values = df.columns.astype(str).values.reshape(-1, 1)

    with h5py.File(output_path, "w") as h5f:

        # matrix
        h5f.create_dataset("data", data=df.values)

        # ----- axis0 (rows) -----
        h5f.create_dataset("axis0", data=np.array(row_values, dtype="S"))
        h5f.create_dataset("levels_axis0", data=np.array(row_levels, dtype="S"))

        # ----- axis1 (columns) -----
        h5f.create_dataset("axis1", data=np.array(col_values, dtype="S"))
        h5f.create_dataset("levels_axis1", data=np.array(col_levels, dtype="S"))


def read_dataframe_from_h5(input_path):
    """
    Read a DataFrame from an HDF5 file with:
        data              (matrix)
        axis0             (row index levels)
        axis1             (column index levels)
        levels_axis0      (names of row index levels)
        levels_axis1      (names of column index levels)
    """
    with h5py.File(input_path, "r") as h5f:
        data = h5f["data"][:]
        axis0 = h5f["axis0"][:].astype(str).tolist()
        axis1 = h5f["axis1"][:].astype(str).tolist()
        levels_axis0 = h5f["levels_axis0"][:].astype(str).tolist()
        levels_axis1 = h5f["levels_axis1"][:].astype(str).tolist()

    if levels_axis0 == ['index']:
        axis0_flatten = [ax[0] for ax in axis0]
        row_index = pd.Index(axis0_flatten)
    else:
        axis0_tuples = [tuple(axis0[i]) for i in range(len(axis0))]
        row_index = pd.MultiIndex.from_tuples(
            axis0_tuples, names=levels_axis0
        )

    if levels_axis1 == ['columns']:
        axis1_flatten = [ax[0] for ax in axis1]
        col_index = pd.Index(axis1_flatten)
    else:
        axis1_tuples = [tuple(axis1[i]) for i in range(len(axis1))]
        col_index = pd.MultiIndex.from_tuples(
            axis1_tuples, names=levels_axis1
        )

    return pd.DataFrame(data, index=row_index, columns=col_index)



# Publish capacity
def sort_capacity_unstacked_into_matrix(capacity_unstacked_df, ordered_manifold_labels):
    """
    The capacity value on the diagonal cannot be computed and hence set to 0.
    Args:
        single_fish_capacity_df (pd.DataFrame): The capacity matrix for a single fish.
        ordered_manifold_labels (list): The ordered list of manifold labels (odors).
    Returns:
        pd.DataFrame: Processed capacity matrix with NaN values replaced and ordered correctly.
    """
    first_odor = capacity_unstacked_df.index[0]
    last_odor = capacity_unstacked_df.columns[-1]
    capacity_unstacked_df[first_odor] = np.nan
    capacity_unstacked_df.loc[last_odor] = np.nan

    capacity_unstacked_df.index = pd.Categorical(
        capacity_unstacked_df.index, categories=ordered_manifold_labels, ordered=True
    )
    capacity_unstacked_df = capacity_unstacked_df.sort_index()
    capacity_unstacked_df = capacity_unstacked_df[ordered_manifold_labels]
    # Replace NaN with zero
    capacity_unstacked_df = capacity_unstacked_df.fillna(0)
    capacity_unstacked_df = capacity_unstacked_df + capacity_unstacked_df.T
    return capacity_unstacked_df



def convert_capacity_df_to_matrix(capacity_df, ordered_manifold_labels, exp_list, manifold_lables):
    capdf_unshuffled = capacity_df.xs(False, level='shuffle')
    capcap = capdf_unshuffled['capacity']
    capcap.index = capcap.index.droplevel(['window_0', 'window_1', 'seed'])
    capcap_mean = capcap.groupby(level=['fish_id', 'condition', 'odor1', 'odor2']).mean().unstack('odor2')

    num_fish = len(exp_list)
    num_manifolds = len(manifold_lables)
    simmat = np.zeros((num_fish, num_manifolds, num_manifolds))

    for idx, (exp_name, condition) in enumerate(exp_list):
        single_fish = capcap_mean.xs((exp_name, condition), level=('fish_id', 'condition')).copy()
        single_fish_mat = sort_capacity_unstacked_into_matrix(single_fish, ordered_manifold_labels=manifold_lables)
        simmat[idx, :, :] = single_fish_mat.values

    levels_axis0 = ['fish_id', 'condition']
    levels_axis1 = ['odor']
    levels_axis2 = ['odor']

    axis0 = exp_list
    axis1 = manifold_lables
    axis2 = manifold_lables

    return simmat, axis0, axis1, axis2, levels_axis0, levels_axis1, levels_axis2


def save_capacity_results_as_h5(capacity_file, exp_list, manifold_lables, output_dir):
    """
    Save the capacity results as an HDF5 file.
    Args:
        capacity_file (str): Path to the capacity file.
        exp_list (list): List of experiments.
        manifold_lables (list): List of manifold labels.
        result_dir (str): Directory to save the results.
        capacity_window_tag (str): Tag for the capacity window.
    """
    # Read the capacity DataFrame
    capdf = pd.read_pickle(capacity_file)
    # Convert the capacity DataFrame to a matrix
    simmat, axis0, axis1, axis2, levels_axis0, levels_axis1, levels_axis2 = convert_capacity_df_to_matrix(
        capdf, manifold_lables, exp_list, manifold_lables
    )
    metric_name = 'capacity'
    output_h5file = os.path.join(output_dir, f'{metric_name}.h5')
    with h5py.File(output_h5file, 'w') as h5f:
        h5f.create_dataset('simmat', data=simmat)
        h5f.create_dataset('axis0', data=np.array(axis0, dtype='S'))
        h5f.create_dataset('axis1', data=np.array(axis1, dtype='S'))
        h5f.create_dataset('axis2', data=np.array(axis2, dtype='S'))
        h5f.create_dataset('levels_axis0', data=np.array(levels_axis0, dtype='S'))
        h5f.create_dataset('levels_axis1', data=np.array(levels_axis1, dtype='S'))
        h5f.create_dataset('levels_axis2', data=np.array(levels_axis2, dtype='S'))

