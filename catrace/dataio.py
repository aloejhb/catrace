import os
import numpy as np
import scipy.io as sio
import glob
import pandas as pd

import itertools

import h5py


# plane_num in Python environment start with 0, plane_num in MATLAB files start with 1
# Useful way of deleting a neuron in all trials
# yy = xx.drop(xx.loc[idx[:, :, 0, 2],:].index)


def get_time_trace_file(root_dir, exp_name, plane_num):
    """Get file path of time trace data based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    time_trace_dir = os.path.join(exp_dir, 'time_trace')
    plane_string = 'plane{0:02d}'.format(plane_num + 1)
    plane_dir = os.path.join(time_trace_dir, plane_string)
    trace_file = os.path.join(plane_dir, 'timetrace_roi.mat')
    return trace_file


def get_spike_dir(root_dir, exp_name, plane_num):
    """Get directory of spike prediction based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    spike_root_dir = os.path.join(exp_dir, 'deconvolution')
    plane_string = 'plane{0:02d}'.format(plane_num + 1)
    spike_dir = os.path.join(spike_root_dir, plane_string)
    return spike_dir


def read_trace(trace_file):
    time_trace = sio.loadmat(trace_file)
    trace_dict = {}
    trace_dict['raw_trace'] = np.stack(time_trace['timeTraceMatList'][0])

    trace_dict['odor_list'] = np.stack(time_trace['odorList'][0]).squeeze()
    trace_dict['roi_tag'] = time_trace['roiTagArray'].squeeze()
    if 'odorArraySorted' in time_trace.keys():
        trace_dict['odor_cat'] = np.stack(
            time_trace['odorArraySorted'].squeeze()
        ).squeeze()
    else:
        # output of older version of neuRoi, assume 3 trial
        # TODO deprecate this part for future neuRoi
        num_trial = 3
        trace_dict['odor_cat'] = list(
            itertools.chain.from_iterable(
                itertools.repeat(i, num_trial) for i in trace_dict['odor_list']
            )
        )
    return trace_dict


def read_spike(spike_dir):
    file_list = sorted(glob.glob(os.path.join(spike_dir, 'spike_*.npy')))
    spike_list = []
    for file in file_list:
        spike = np.load(file)
        spike_list.append(spike)
    spike_array = np.stack(spike_list)
    return spike_array


def convert_trial_to_df(trial_trace, roi_tags):
    trial_df = pd.DataFrame(trial_trace, index=roi_tags)
    return trial_df


def load_trace_file(root_dir, exp_name, plane_nb_list, num_trial, odor_list):
    df_list = [None] * len(plane_nb_list)
    for i, plane_nb in enumerate(plane_nb_list):
        trace_file = get_time_trace_file(root_dir, exp_name, plane_nb)
        trace_dict = read_trace(trace_file)
        odordf = pd.DataFrame(trace_dict['odor_cat'], columns=['odor_cat'])
        odordfi = (
            odordf.groupby('odor_cat')
            .apply(lambda x: x.reset_index())
            .sort_values('index')
        )
        index = odordfi.index
        trial_df_list = [
            convert_trial_to_df(t, trace_dict['roi_tag'])
            for t in trace_dict['raw_trace']
        ]
        df_list[i] = pd.concat(
            trial_df_list, keys=index, names=['odor', 'trial', 'neuron']
        )
    tracedf = pd.concat(
        df_list, keys=plane_nb_list, names=['plane'] + df_list[0].index.names
    )
    tracedf = tracedf.reorder_levels(['odor', 'trial', 'plane', 'neuron'])
    tracedf = tracedf.reindex(odor_list, level='odor')
    return tracedf


def load_spike_file(root_dir, exp_name, plane_num):
    spike_dir = get_spike_dir(root_dir, exp_name, plane_num)
    spike_array = read_spike(spike_dir)
    return spike_array


def load_experiment(root_dir, exp_subdir, exp_name=None):
    if exp_name is None:
        exp_name = exp_subdir
    exp_file_name = 'experimentConfig_{}.mat'.format(exp_name)
    exp_file = os.path.join(root_dir, exp_subdir, exp_file_name)
    foo = sio.loadmat(exp_file)
    myexp_struct = foo['myexpStruct']

    exp_info = {}
    exp_info_struct = myexp_struct[0][0][0][0][0]
    exp_info['frame_rate'] = exp_info_struct[1][0, 0]
    exp_info['odor_list'] = list(np.concatenate(exp_info_struct[2][0]))
    exp_info['num_trial'] = exp_info_struct[3][0, 0]
    exp_info['num_plane'] = exp_info_struct[4][0, 0]
    return exp_info


def tracedf_to_hdf5_data(dff: pd.DataFrame):
    """
    Extracts arrays and metadata from the MultiIndex DataFrame exactly as in the notebook.
    """
    axis0 = dff.index.droplevel('time').unique().values
    axis1 = dff.index.get_level_values('time').unique().values
    axis2 = dff.columns.values

    levels_axis0 = dff.index.droplevel('time').names
    levels_axis1 = ['frame']
    levels_axis2 = dff.columns.names

    num_trials = len(axis0)
    num_frames = len(axis1)
    num_rois = len(axis2)

    traces = np.empty((num_trials, num_frames, num_rois))
    for i, trial_key in enumerate(axis0):
        traces[i, :, :] = dff.xs(trial_key, level=('odor', 'trial')).values

    return traces, axis0, axis1, axis2, levels_axis0, levels_axis1, levels_axis2


def save_tracedf_to_hdf5(dff: pd.DataFrame, filepath: str):
    """
    Saves the MultiIndex DataFrame to HDF5 exactly as in the notebook.
    """
    traces, axis0, axis1, axis2, levels_axis0, levels_axis1, levels_axis2 = tracedf_to_hdf5_data(dff)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('traces', data=traces, compression='gzip')
        f.create_dataset('axis0', data=np.array(axis0.tolist(), dtype='S'))
        f.create_dataset('axis1', data=np.array(axis1.tolist()))
        f.create_dataset('axis2', data=np.array(axis2.tolist()))
        f.create_dataset('levels_axis0', data=np.array(levels_axis0, dtype='S'))
        f.create_dataset('levels_axis1', data=np.array(levels_axis1, dtype='S'))
        f.create_dataset('levels_axis2', data=np.array(levels_axis2, dtype='S'))


def load_tracedf_from_hdf5(filepath: str) -> pd.DataFrame:
    """
    Loads the HDF5 file back into the original MultiIndex DataFrame exactly as in the notebook.
    """
    with h5py.File(filepath, 'r') as f:
        traces = f['traces'][:]
        axis0 = f['axis0'][:].astype(str)
        axis1 = f['axis1'][:]
        axis2 = f['axis2'][:]
        levels_axis0 = f['levels_axis0'][:].astype(str)
        levels_axis1 = f['levels_axis1'][:].astype(str)
        levels_axis2 = f['levels_axis2'][:].astype(str)

    dfs = []
    for i, trial_key in enumerate(axis0):
        odor, trial = trial_key
        trial = int(trial)  # convert trial to int if needed
        df_temp = pd.DataFrame(traces[i], index=axis1)
        df_temp.index = pd.MultiIndex.from_product(
            [[odor], [trial], df_temp.index],
            names=list(levels_axis0) + list(levels_axis1)
        )
        dfs.append(df_temp)

    dff_reconstructed = pd.concat(dfs)
    if len(axis2.shape) == 1:
        # If axis2 is a single level, we can use it directly
        dff_reconstructed.columns = pd.Index(axis2, name=levels_axis2[0])
    else:
        # If axis2 is multi-level, we need to create a MultiIndex
        dff_reconstructed.columns = pd.MultiIndex.from_arrays(axis2.T, names=levels_axis2)
    return dff_reconstructed
