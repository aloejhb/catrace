import os
import numpy as np
import scipy.io as sio
import glob
import pandas as pd
import re




def get_spike_dir(root_dir, exp_name, plane_num):
    """Get directory of spike prediction based on the directory structure of neuRoi"""
    exp_dir = os.path.join(root_dir, exp_name)
    spike_root_dir = os.path.join(exp_dir, 'deconvolution')
    plane_string = 'plane{0:02d}'.format(plane_num)
    spike_dir = os.path.join(spike_root_dir, plane_string)
    return spike_dir


def read_spike(spike_dir):
    file_list = sorted(glob.glob(os.path.join(spike_dir, 'spike_*.npy')))
    spike_list = []
    for file in (file_list):
        spike = np.load(file)
        spike_list.append(spike)
    spike_array = np.stack(spike_list)
    return spike_array


def load_spike_file(root_dir, exp_name, plane_num):
    spike_dir = get_spike_dir(root_dir, exp_name, plane_num)
    spike_array = read_spike(spike_dir)
    return spike_array


def load_experiment(root_dir, exp_name):
    exp_file_name = 'experimentConfig_struct_{}.mat'.format(exp_name)
    exp_file = os.path.join(root_dir, exp_name, exp_file_name)
    foo = sio.loadmat(exp_file)
    myexp_struct = foo['myexpStruct']

    myexp = {}
    exp_info = {}
    exp_info_struct = myexp_struct[0][0][0][0][0]
    exp_info['frame_rate'] = exp_info_struct[1][0, 0]
    exp_info['odor_list'] = list(np.concatenate(exp_info_struct[2][0]))
    exp_info['num_trial'] = exp_info_struct[3][0, 0]
    exp_info['num_plane'] = exp_info_struct[4][0, 0]
    myexp['exp_info'] = exp_info
    myexp['raw_file_list'] = list(np.concatenate(np.concatenate(myexp_struct[0][0][2])))
    myexp['exp_dir'] = myexp_struct[0][0][3][0]
    return myexp


def get_odor_name(file_name):
    match = re.match('.*s\d_o\d([a-zA-Z]+)_.*', file_name)
    if match:
        odor = match.group(1)
    else:
        odor = None
    return odor


def generate_exp_dataframe(raw_file_list, odor_list):
    raw_odor_list = [get_odor_name(x) for x in raw_file_list]
    expdf = pd.DataFrame(list(raw_file_list), columns=['file'])
    raw_odor_cat = pd.Categorical(raw_odor_list, categories=odor_list,
                                  ordered=True)
    expdf['odor'] = raw_odor_cat
    expdf = expdf.sort_values(by=['odor'])
    return expdf


def read_trace(trace_file):
    time_trace = sio.loadmat(trace_file)
    import pdb; pdb.set_trace()

    trace_dict = {}
    trace_dict['raw_trace'] = np.stack(time_trace['timeTraceMatList'][0])
    trace_dict['df_trace'] = np.stack(time_trace['timeTraceDfMatList'][0])
    trace_dict['odor_list'] = np.stack(time_trace['odorList'][0]).squeeze()
    return trace_dict


def get_time_trace_file(exp_dir, plane_nb, file_name):
    """Get file path of time trace data based on the directory structure of neuRoi"""
    time_trace_dir = os.path.join(exp_dir, 'time_trace')
    plane_string = 'plane{0:02d}'.format(plane_nb)
    plane_dir = os.path.join(time_trace_dir, plane_string)
    trace_file_regexp = os.path.join(plane_dir, file_name.replace('.tif', '*.mat'))
    trace_file = glob.glob(trace_file_regexp)
    if trace_file:
        trace_file = trace_file[0]
    else:
        trace_file = None
    return trace_file


def load_trace_file(exp_dir, plane_nb, file_name):
    trace_file = get_time_trace_file(exp_dir, plane_nb, file_name)
    trace_dict = read_trace(trace_file)
    return trace_dict


def load_trace_to_df(exp, plane_nb, keep_common_roi=True):
    trace_dict_list = [load_trace_file(exp['exp_dir'], plane_nb, x)
                       for x in exp['dataframe']['file']]
    roi_list = [x['roiArray'] for x in trace_dict_list]
    trace_list = [x['timeTraceMat'] for x in trace_dict_list]
    # if keep_common_roi:
        
    colname = 'trace_plane{0:02d}'.format(plane_nb)
    exp['dataframe'][colname] = trace_list
