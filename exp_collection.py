import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition

from . import pattern_correlation as pcr
from . import process_time_trace as ptt


def plot_exp_pattern_correlation(dfovf_cut, odor_list, frame_rate,
                                 time_window=[5.3,7.3], ax=None):
    time_window = np.array(time_window)
    fwindow = ptt.convert_sec_to_frame(time_window, frame_rate)
    dfrestack = ptt.restack_as_pattern(dfovf_cut.iloc[:, fwindow[0]:fwindow[1]])
    pat = dfrestack.mean(level=[0, 1])

    pat = pat.reset_index()
    pat['odor'] = pd.Categorical(pat.odor, ordered=True,
                                 categories=odor_list)
    pat = pat.sort_values('odor')

    if ax is None:
        fig, ax = plt.subplots()
    im = pcr.plot_pattern_correlation(pat, ax, clim=(0, 1))



# def load_dfovf(data_root_dir, exp_name, region_name):
#     # Load data
#     exp_subdir = os.path.join(exp_name, region_name)
#     tracedf = dataio.load_trace_file(data_root_dir, exp_subdir, plane_nb_list, num_trial, odor_list)

#     # Cut first X second to exclude PMT off period
#     cut_time = 5+2
#     cut_win = convert_sec_to_frame([cut_time, 40], frame_rate)
#     tracedf = ptt.cut_tracedf(tracedf, cut_win[0], 0, cut_win[1])

#     # Calculate dF/F
#     fzero_twindow = np.array([7, 9])
#     dfovf = ptt.compute_dfovf(tracedf, fzero_twindow, frame_rate, intensity_offset=-10)

#     # Detect or set response onset
#     xwindow = ((np.array([15, 20])-cut_time)*frame_rate).astype('int')
#     onset_param = dict(thresh=0.013, sigma=3, xwindow=xwindow)
#     dfovf_avg, y, dy = ptt.detect_tracedf_onset(dfovf, onset_param, debug=True)
#     dfovf_avg['onset'] = 70
#     # dfovf_avg['onset'] = dfovf_avg['onset'].astype('int32')


#     # Align time traces to onset
#     pre_time = 4
#     post_time = 18
#     dfovf_cut = ptt.align_tracedf(dfovf, dfovf_avg['onset'],
#                                   pre_time, post_time, frame_rate)
#     return dfovf_cut

def plot_explist(data_list, plot_func, sharex, sharey, *args, **kwargs):
        ncol = 5
        nrow = int(np.ceil(len(data_list) / ncol))
        figsize=[17, 3.4*nrow]
        fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                                 sharey=sharey, figsize=figsize)
        for idx, data in enumerate(data_list):
            ax = axes.flatten()[idx]
            plot_func(data, *args, **kwargs, ax=ax)
        plt.tight_layout()
        return fig


def plot_explist_decorator(plot_func, csplus_dict, data_dict, sharex=False, sharey=False):
    def plot_explist_wrapper(csplus, region, *args, **kwargs):
        csexp_list = csplus_dict[csplus]
        data_list = [data_dict[exp][region] for exp in csexp_list]
        return plot_explist(data_list, plot_func, sharex, sharey, *args, **kwargs)
    return plot_explist_wrapper



def get_data_dict_decorator(exp_list, region_list, dfovf_dict, data_func):
    def get_data_dict(*args, **kwargs):
        data_dict = dict()
        for exp in exp_list:
            exp_name = exp[0]
            data_dict[exp_name] = dict()
            for region in region_list:
                print(exp_name, region)
                dfovf = dfovf_dict[exp_name][region]
                data_dict[exp_name][region] = data_func(dfovf,
                                                        *args, **kwargs)
        return data_dict
    return get_data_dict
