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


def plot_explist_decorator(plot_func, data_dict, sharex=True, sharey=True):
    def plot_explist(csplus, region, *args, **kwargs):
        csexp_list = csplus_dict[csplus]
        ncol = 5
        nrow = int(np.ceil(len(csexp_list) / ncol))
        if nrow == 2:
            figsize=[17, 6.7]
        else:
            figsize=[17, 3.4]
        fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                                 sharey=sharey, figsize=figsize)
        for idx,csexp in enumerate(csexp_list):
            data = data_dict[csexp][region]
            ax = axes.flatten()[idx]
            plot_func(data, *args, **kwargs, ax=ax)
        plt.tight_layout()
        return fig
    return plot_explist


def compute_pca(dfovf, n_components, tbin=5):
    # Bin time traces over time
    dfovf_bin = ptt.bin_tracedf(dfovf, tbin)
    pattern = ptt.restack_as_pattern(dfovf_bin)
    pca = decomposition.PCA(n_components)
    latent = pca.fit_transform(pattern)
    results = dict(latent= latent, index=pattern.index)
    return results


def plot_embed_2d(results, component_idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    embeddf = pd.DataFrame(results['latent'][:,component_idx],
                           columns=['x', 'y'],
                           index=results['index'])
    groups = embeddf.groupby(['odor'])
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='-', ms=4, label=name, alpha=0.7)



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
