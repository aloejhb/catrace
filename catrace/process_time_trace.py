import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from .frame_time import convert_sec_to_frame
from .import process_time_trace as ptt


def compute_dfovf(trace, fzero_twindow, frame_rate=1, intensity_offset=0,
                  fzero_percent=0.5):
    # Calculate dF/F from raw time traces
    # trace: a NxM matrix/dataframe containing N traces of length M
    fzero_window = convert_sec_to_frame(fzero_twindow, frame_rate)
    trace_fg = trace - intensity_offset
    if isinstance(trace_fg, pd.DataFrame):
        trace_zero = trace_fg.iloc[:, fzero_window[0]:fzero_window[1]]
    else:
        trace_zero = trace_fg[:, fzero_window[0]:fzero_window[1]]
    fzero = np.quantile(trace_zero, fzero_percent, axis=1)
    dfovf = (trace_fg - fzero[:, None]) / fzero[:, None]
    return dfovf


def detect_onset(y, thresh, xwindow, sigma=0, normalize=True, debug=False):
    if sigma:
        y = gaussian_filter1d(y, sigma)
    if normalize:
        miny = min(y)
        maxy = max(y)
        y = (y - miny)/(maxy - miny)
    dy = np.gradient(y)
    xvec = np.arange(len(y))
    onset = xvec[(dy > thresh) & (xvec >= xwindow[0]) & (xvec <=xwindow[1])]
    if len(onset) == 0:
        onset = np.nan
    else:
        onset = onset[0]

    if debug:
        return onset, y, dy
    else:
        return onset


def detect_tracedf_onset(trace, onset_param, debug=False):
    trial_avg = trace.groupby(level=['odor', 'trial']).mean()
    if debug:
        onset_param['debug'] = True
    onset_list = [detect_onset(row, **onset_param) for index, row in trial_avg.iterrows()]
    if debug:
        on_list = [x[0] for x in onset_list]
        y = np.array([x[1] for x in onset_list])
        dy = np.array([x[2] for x in onset_list])
        trial_avg['onset'] = on_list
        return trial_avg, y, dy
    else:
        trial_avg['onset'] = onset_list
        return trial_avg


def cut_tracedf(tracedf, onset, pre_nframe, post_nframe):
    cdf = tracedf.iloc[:, onset-pre_nframe:onset+post_nframe]
    cdf.columns = pd.RangeIndex(start=0, stop=len(cdf.columns), step=1)
    return cdf


def align_tracedf(tracedf, onsetdf, pre_time, post_time, frame_rate):
    pre_nframe = int(pre_time * frame_rate)
    post_nframe = int(post_time * frame_rate)
    tracedf_group = tracedf.groupby(['odor', 'trial'])
    cut_df = [cut_tracedf(group, onsetdf.loc[name],
                          pre_nframe, post_nframe)
              for name, group in tracedf_group]
    cut_df = pd.concat(cut_df)
    return cut_df

def select_response(tracedf, snr_thresh, base_window, response_window, frame_rate):
    base_fwindow = convert_sec_to_frame(base_window, frame_rate)
    response_fwindow = convert_sec_to_frame(response_window, frame_rate)
    base = tracedf.loc[:,base_fwindow[0]:base_fwindow[1]]
    response = tracedf.loc[:,response_fwindow[0]:response_fwindow[1]]
    tracedf['response'] = (response.max(axis=1) > snr_thresh[0]*base.std(axis=1)) & (response.max(axis=1) < snr_thresh[1]*base.std(axis=1))

    select = tracedf['response'].groupby(level=[2,3]).max()
    tracedf = tracedf.merge(select,left_on=('plane','neuron'),
                            left_index=True,right_on=('plane','neuron'),
                            right_index=True)
    tracedf = tracedf[tracedf['response_y']].drop(columns=['response_x','response_y'])
    return tracedf


def bin_tracedf(tracedf, bin_factor):
    bindf = tracedf.groupby(np.arange(len(tracedf.columns))//bin_factor, axis=1).mean()
    return bindf


def restack_as_pattern(tracedf):
    newdf = tracedf.stack()
    newdf.index = newdf.index.rename(newdf.index.names[0:-1]+['time'])
    index = newdf.index
    newdf = newdf.unstack(['plane', 'neuron'])
    newdf = newdf.reindex(index.unique('odor'), level='odor')
    return newdf


def bin_and_restack(dfovf, tbin):
    dfovf_bin = bin_tracedf(dfovf, tbin)
    pattern = restack_as_pattern(dfovf_bin)
    return pattern


def select_neuron(dfovf, thresh):
    dfovf_restack = ptt.restack_as_pattern(dfovf)
    deviation = (dfovf_restack - dfovf_restack.mean()).abs().max() \
        / dfovf_restack.std()
    idx = deviation >= thresh
    dfovf_select_restack = dfovf_restack.loc[:,idx]
    dfovf_select = dfovf_select_restack.stack(level=['plane','neuron']).unstack(level='time')
    return dfovf_select
