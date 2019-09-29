import os
import numpy as np
import scipy.io as sio
import glob
import pandas as pd


def get_time_trace_file(root_dir, exp_name, plane_num):
    """Get file path of time trace data based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    time_trace_dir = os.path.join(exp_dir, 'time_trace')
    plane_string = 'plane{0:02d}'.format(plane_num)
    plane_dir = os.path.join(time_trace_dir, plane_string)
    trace_file = os.path.join(plane_dir, 'timetrace.mat')
    return trace_file


def get_spike_dir(root_dir, exp_name, plane_num):
    """Get directory of spike prediction based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    spike_root_dir = os.path.join(exp_dir, 'deconvolution')
    plane_string = 'plane{0:02d}'.format(plane_num)
    spike_dir = os.path.join(spike_root_dir, plane_string)
    return spike_dir


def read_trace(trace_file):
    time_trace = sio.loadmat(trace_file)
    trace_dict = {}
    trace_dict['raw_trace'] = np.stack(time_trace['timeTraceMatList'][0])
    trace_dict['odor_cat'] = np.stack(time_trace['odorArraySorted'].squeeze()).squeeze()
    # trace_dict['df_trace'] = np.stack(time_trace['timeTraceDfMatList'][0])
    trace_dict['odor_list'] = np.stack(time_trace['odorList'][0]).squeeze()
    return trace_dict


def read_spike(spike_dir):
    file_list = sorted(glob.glob(os.path.join(spike_dir, 'spike_*.npy')))
    spike_list = []
    for file in (file_list):
        spike = np.load(file)
        spike_list.append(spike)
    spike_array = np.stack(spike_list)
    return spike_array


def load_trace_file(root_dir, exp_name, plane_nb_list):
    for i, plane_nb in enumerate(plane_nb_list):
        trace_file = get_time_trace_file(root_dir, exp_name, plane_nb)
        trace_dict = read_trace(trace_file)
        trace_colname = 'raw_trace_plane{0:02d}'.format(plane_nb)
        if i == 0:
            tracedf = pd.DataFrame(list(zip(trace_dict['odor_cat'],
                                            trace_dict['raw_trace'])),
                                   columns=['odor', trace_colname])
            cat_dtype = pd.api.types.CategoricalDtype(categories=trace_dict['odor_list'], ordered=True)
            tracedf['odor'] = tracedf['odor'].astype(cat_dtype)
            tracedf['odor_code'] = tracedf['odor'].cat.codes
            col_titles = ['odor_code', 'odor', trace_colname]
            tracedf = tracedf.reindex(columns=col_titles)
        else:
            tracedf[trace_colname] = list(trace_dict['raw_trace'])
    return tracedf


def load_spike_file(root_dir, exp_name, plane_num):
    spike_dir = get_spike_dir(root_dir, exp_name, plane_num)
    spike_array = read_spike(spike_dir)
    return spike_array


def load_experiment(root_dir, exp_name):
    exp_file_name = 'experimentConfig_struct_{}.mat'.format(exp_name)
    exp_file = os.path.join(root_dir, exp_name, exp_file_name)
    foo = sio.loadmat(exp_file)
    myexp_struct = foo['myexpStruct']

    exp_info = {}
    exp_info_struct = myexp_struct[0][0][0][0][0]
    exp_info['frame_rate'] = exp_info_struct[1][0, 0]
    exp_info['odor_list'] = list(np.concatenate(exp_info_struct[2][0]))
    exp_info['num_trial'] = exp_info_struct[3][0, 0]
    exp_info['num_plane'] = exp_info_struct[4][0, 0]
    return exp_info
