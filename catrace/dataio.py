import os
import numpy as np
import scipy.io as sio
import glob
import pandas as pd

import itertools

# plane_num in Python environment start with 0, plane_num in MATLAB files start with 1
# Useful way of deleting a neuron in all trials
# yy = xx.drop(xx.loc[idx[:, :, 0, 2],:].index)

def get_time_trace_file(root_dir, exp_name, plane_num):
    """Get file path of time trace data based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    time_trace_dir = os.path.join(exp_dir, 'time_trace')
    plane_string = 'plane{0:02d}'.format(plane_num+1)
    plane_dir = os.path.join(time_trace_dir, plane_string)
    trace_file = os.path.join(plane_dir, 'timetrace.mat')
    return trace_file


def get_spike_dir(root_dir, exp_name, plane_num):
    """Get directory of spike prediction based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    spike_root_dir = os.path.join(exp_dir, 'deconvolution')
    plane_string = 'plane{0:02d}'.format(plane_num+1)
    spike_dir = os.path.join(spike_root_dir, plane_string)
    return spike_dir


def read_trace(trace_file):
    time_trace = sio.loadmat(trace_file)
    trace_dict = {}
    trace_dict['raw_trace'] = np.stack(time_trace['timeTraceMatList'][0])

    trace_dict['odor_list'] = np.stack(time_trace['odorList'][0]).squeeze()
    if 'odorArraySorted' in time_trace.keys():
        trace_dict['odor_cat'] = np.stack(time_trace['odorArraySorted'].squeeze()).squeeze()
    else:
        # output of older version of neuRoi, assume 3 trial
        # TODO deprecate this part for future neuRoi
        num_trial = 3
        trace_dict['odor_cat'] = list(itertools.chain.from_iterable(itertools.repeat(i, num_trial) for i in trace_dict['odor_list']))
    return trace_dict


def read_spike(spike_dir):
    file_list = sorted(glob.glob(os.path.join(spike_dir, 'spike_*.npy')))
    spike_list = []
    for file in (file_list):
        spike = np.load(file)
        spike_list.append(spike)
    spike_array = np.stack(spike_list)
    return spike_array


def load_trace_file(root_dir, exp_name, plane_nb_list, num_trial, odor_list):
    df_list = [None] * len(plane_nb_list)
    for i, plane_nb in enumerate(plane_nb_list):
        trace_file = get_time_trace_file(root_dir, exp_name, plane_nb)
        trace_dict = read_trace(trace_file)
        # trace_colname = 'raw_trace_plane{0:02d}'.format(plane_nb)
        # index_iter = [trace_dict['odor_list'], np.arange(num_trial)]
        # index = pd.MultiIndex.from_product(index_iter)
        odordf = pd.DataFrame(trace_dict['odor_cat'],columns=['odor_cat'])
        odordfi = odordf.groupby('odor_cat').apply(lambda x: x.reset_index()).sort_values('index')
        index = odordfi.index
        df_list[i] = pd.concat([pd.DataFrame(x) for x in trace_dict['raw_trace']],
                               keys=index, names=['odor', 'trial', 'neuron'])
    tracedf = pd.concat(df_list, keys=plane_nb_list,
                        names=['plane']+df_list[0].index.names)
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
