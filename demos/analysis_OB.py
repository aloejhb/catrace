import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
from sklearn import manifold
from sklearn import decomposition
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../')
from catrace import dataio
import catrace.process_time_trace as ptt
import catrace.plot_trace as pltr
import catrace.pattern_correlation as pcr
import catrace.manifold_embed as emb
from catrace.frame_time import convert_sec_to_frame
from catrace.trace_dataframe import concatenate_planes


if __name__ == '__main__':
    reload(dataio)
    reload(pltr)
    reload(pcr)
    figs = {}
    data_root_dir = '<root directory for data>'
    exp_name = '2021-04-02-DpOBEM-JH11'
    region_name = 'OB'

    outdir = os.path.join(data_root_dir, exp_name, region_name, 'analysis')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Load data
    exp_subdir = os.path.join(exp_name, region_name)
    # exp_info = dataio.load_experiment(data_root_dir, exp_subdir, exp_name=exp_name)
    # frame_rate = exp_info['frame_rate'] / exp_info['num_plane']
    # plane_nb_list = range(1, 5)
    frame_rate = 30/4
    num_trial = 3
    exp_info = dict(num_trial=3)
    # odor_list = ['ala', 'trp', 'ser', 'tdca', 'tca', 'gca', 'acsf', 'spont']
    odor_list = ['phe', 'trp', 'arg', 'tdca', 'tca', 'gca', 'acsf', 'spont']
    # plane_nb_list = np.array([1,2,3,4]) - 1
    # plane_nb_list = np.array([3]) - 1
    tracedf = dataio.load_trace_file(data_root_dir, exp_subdir, plane_nb_list, num_trial, odor_list)
    # Drop neuron 33 in plane 4, since its dF/F is inf
    # idx = pd.IndexSlice
    # tracedf.drop(tracedf.loc[idx[:, :, 3, 33],:].index, inplace=True)

    # Cut first X second to exclude PMT off period
    cut_time = 5
    cut_win = convert_sec_to_frame([cut_time, 40], frame_rate)
    tracedf = ptt.cut_tracedf(tracedf, cut_win[0], 0, cut_win[1])


    # Calculate dF/F
    fzero_twindow = np.array([11.5, 12.5])
    dfovf = ptt.compute_dfovf(tracedf, fzero_twindow, frame_rate, intensity_offset=-40)

    # # Plot time trace heatmap
    # num_odor = len(odor_list)
    # climit = (-0.1, 1.3)
    # cut_window = (np.array([10, 35]) * frame_rate).astype('int')
    # # pltr.plot_tracedf_heatmap(tracedf, exp_info['num_trial'], num_odor, plane_nb_list, cut_window, climit)
    # pltr.plot_tracedf_heatmap(dfovf, exp_info['num_trial'], odor_list, climit, cut_window)
    # plt.show()

    # Detect response onset
    plotfig = True
    xwindow = ((np.array([15, 20])-cut_time)*frame_rate).astype('int')
    onset_param = dict(thresh=0.013, sigma=3, xwindow=xwindow)
    # onset_param_list = [onset_param] * len(plane_nb_list)
    # dfovf_avg['onset'] is the onset in frame number
    dfovf_avg, y, dy = ptt.detect_tracedf_onset(dfovf, onset_param, debug=True)
    print(dfovf_avg)
    dfovf_avg['onset'] = 92
    dfovf_avg['onset'] = dfovf_avg['onset'].astype('int32')
    if plotfig:
        figs['detect_onset'], axes = plt.subplots(2, 1, sharex=True)
        plot_win = ((np.array([10, 40])-cut_time)*frame_rate).astype('int')
        lines = axes[0].plot(y[0:20, plot_win[0]:plot_win[1]].transpose())
        [axes[0].vlines(x-plot_win[0], 0, 2, color=lines[i].get_color())
         for i, x in enumerate(dfovf_avg['onset'][0:20])]
        axes[0].set_ylabel('y')
        axes[1].plot(dy[0:20, plot_win[0]:plot_win[1]].transpose())
        axes[1].hlines(onset_param['thresh'], 0, 150)
        axes[1].set_ylabel('dy')
        axes[1].set_xlabel('# frame')
        # plt.xlim(xwindow)
        # plt.ylim(-0.01, 1.8)

        fig_file = os.path.join(outdir, 'detect_onset.svg')
        plt.savefig(fig_file)

    # Align time traces to onset
    pre_time = 5
    post_time = 18
    dfovf_cut = ptt.align_tracedf(dfovf, dfovf_avg['onset'], pre_time, post_time, frame_rate)

    # Select responsive neurons
    # TODO remove or modify this !!!!
    # base_window = [0, 4]
    # response_window = [6, 10]
    # snr_thresh = [4, 10]
    # dfovf_cut = ptt.select_response(dfovf_cut, snr_thresh, base_window, response_window, frame_rate)


    # Plot average time trace
    plt_avg = True
    if plt_avg:
        figs['avg_cut'] = pltr.plot_trace_avg(dfovf_cut, odor_list, frame_rate)
        plt.xlabel('Time (s)')
        fig_file = os.path.join(outdir, 'avg_cut.svg')
        plt.savefig(fig_file)



    # Bin time traces over time
    dfovf_bin = ptt.bin_tracedf(dfovf_cut, 4)

    pattern = ptt.restack_as_pattern(dfovf_bin)

    climit = (-0.1, 1.5)
    if plotfig:
        figs['heatmap_cut'] = pltr.plot_tracedf_heatmap(dfovf_bin, 3, odor_list, climit)
        fig_file = os.path.join(outdir, 'heatmap_cut.svg')
        plt.savefig(fig_file)


    plane_nb = plane_nb_list[0]
    pattern_plane = pattern
    num_neighbor = 10
    # n_components = 2
    # X_iso = manifold.Isomap(num_neighbor, n_components=n_components).fit_transform(pattern_plane)
    n_components = 3
    pca = decomposition.PCA(n_components)
    X_iso = pca.fit_transform(pattern_plane)
    if n_components == 2:
        embeddf = pd.DataFrame(X_iso, columns=['x', 'y'], index=pattern.index)
    else:
        embeddf = pd.DataFrame(X_iso, columns=['x', 'y', 'z'], index=pattern.index)
    figs['embed'] = emb.plot_embed(embeddf)
    method = 'pca'
    fig_file = os.path.join(outdir, method+'_plane{0:d}.svg'.format(plane_nb+1))
    plt.savefig(fig_file)



    # Plot average time trace
    # plane_nb_list = [4]
    # pltr.plot_trace_avg(dfovf, plane_nb_list, frame_rate)

    # Calculate response pattern
    # plane_nb_list = [4]
    # time_window = [5, 10]
    # trace = np.stack(concatenate_planes(tracedf, plane_nb_list))
    # pat = pcr.convert_trace_to_pattern(trace, time_window, frame_rate)



    # # Plot correlation matrix
    time_window = np.array([6,10])
    fwindow = ptt.convert_sec_to_frame(time_window, frame_rate)
    dfrestack = ptt.restack_as_pattern(dfovf_cut.iloc[:, fwindow[0]:fwindow[1]])
    pat = dfrestack.mean(level=[0, 1])

    pat = pat.reset_index()
    pat['odor'] = pd.Categorical(pat.odor, ordered=True,
                                 categories=odor_list)
    pat = pat.sort_values('odor')
    figs['pattern_correlation'] = plt.figure()
    im = pcr.plot_pattern_correlation(pat, plt.gca(), clim=(0, 1))

    fig_file = os.path.join(outdir, 'pattern_correlation.svg')
    plt.savefig(fig_file)

    # # Plot decorrelation
    # # plane_nb_list = [3]
    # trace = np.stack(concatenate_planes(tracedf, plane_nb_list))
    # pcr.plot_decorrelation(trace, plt.gca(), frame_rate=frame_rate, perc=50, time_window=time_window)
    # plt.show()


    # rank_order=[4, 18, 24,  5,  9, 13,  3,  7, 12, 11, 19, 20,  2, 8, 17,  1,  6, 14, 10, 15, 16, 21, 22, 23]
    # pat2 = pat
    # pat2['orig_idx'] = rank_order
    # pat2 = pat.sort_values('orig_idx')
    # pat = pat.drop(columns='orig_idx')
    # pat2 = pat2.drop(columns='orig_idx')
    # plt.figure()
    # im = pcr.plot_pattern_correlation(pat2, plt.gca(), clim=(0, 0.5))

    # fig_file = os.path.join(outdir, 'pattern_correlation_time_order.svg')
    # plt.savefig(fig_file)

    report_file = os.path.join(outdir, 'analysis_report.pdf')
    with PdfPages(report_file) as pdf:
        for fig in figs.values():
            pdf.savefig(fig)
