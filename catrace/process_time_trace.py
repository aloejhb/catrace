import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .frame_time import convert_sec_to_frame
from . import process_time_trace as ptt
from . import frame_time
from . import utils

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


def cut_tracedf(tracedf, window):
    cdf = tracedf.iloc[:, window[0]:window[1]]
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


def restack_as_pattern(df):
    df.columns = df.columns.set_names('time')
    newdf = df.unstack(['plane', 'neuron']).stack('time')
    # newdf = tracedf.stack()
    # newdf.index = newdf.index.rename(newdf.index.names[0:-1]+['time'])
    # index = newdf.index
    # newdf = newdf.reindex(index.unique('odor'), level='odor')
    return newdf


def unstack_pattern(df):
    return df.transpose().stack(level=['odor', 'trial'])


def bin_tracedf(tracedf, bin_size, axis=0):
    if axis == 1:
        tracedf = restack_as_pattern(tracedf)# tracedf.transpose()

    time_index = tracedf.index.get_level_values(level='time')

    bins = np.arange(0, len(time_index.unique())+bin_size, bin_size) - 1
    tracedf['time_bin'] = pd.cut(time_index, bins)
    binned_dfovf = tracedf.set_index('time_bin', append=True)

    names = list(binned_dfovf.index.names)
    names.remove('time')
    binned_dfovf = binned_dfovf.groupby(level=names).mean()
    binned_dfovf = binned_dfovf.reindex(tracedf.index.unique('odor'), level='odor')

    return binned_dfovf


def truncate_binned_df(df, frame_window):
    frame_interval = pd.Interval(left=frame_window[0], right=frame_window[1])
    selected_index = [tb.overlaps(frame_interval) for tb in df.index.get_level_values('time_bin')]
    df_trunc = df[selected_index]
    return df_trunc


def truncate_tracedf(df, frame_window):
    df = restack_as_pattern(df)
    times = df.index.get_level_values('time')
    df_trunc = df[(times >= frame_window[0]) & (times <= frame_window[1])]
    return df_trunc


def truncate_df_window(df, frame_window):
    if 'time_bin' in df.index.names:
        df_trunc = truncate_binned_df(df, frame_window)
    else:
        df_trunc = truncate_tracedf(df, frame_window)
    return df_trunc


def mean_pattern_in_time_window(df, time_window, frame_rate):
    """Compute pattern of neuron responses averaged within a time window"""
    time_window = np.array(time_window)
    fwindow = frame_time.convert_sec_to_frame(time_window, frame_rate)
    df_filtered = df[(df.index.get_level_values('time') >= fwindow[0])
                     & (df.index.get_level_values('time') <= fwindow[1])]
    all_levels = list(df.index.names)
    all_levels.remove('time')
    # Group by except time, sort=False keeps the original order
    pattern = df_filtered.groupby(level=all_levels, sort=False).mean()
    return pattern


def compute_deviation(dfovf, std_window=None, sigma=None):
    if std_window is not None:
        std = dfovf[std_window[0]:std_window[1]].std()
    else:
        std = dfovf.std()

    if sigma:
        dfovf_filtered = gaussian_filter1d(dfovf, sigma, axis=0)
        dfovf_filtered = utils.copy_frame_structure(dfovf_filtered, dfovf)
    else:
        dfovf_filtered = dfovf

    deviation = (dfovf_filtered - dfovf_filtered.mean()).abs().max() / std
    return deviation

def compute_max_of_mean_response_per_trial(df, response_window,
                                           normalize_by_std=False,
                                           std_window=None):
    df = select_time_points(df, response_window)
    response = df.groupby(level=['odor', 'trial']).mean().max()

    if normalize_by_std:
        std = dfovf[std_window[0]:std_window[1]].std()
        response = response / std

    return response

@dataclass_json
@dataclass
class SelectNeuronsConfig:
    response_window: list[int]

    thresh: float = None
    head: int = None

    normalize_by_std: bool = False
    std_window: list[int] = None


def get_select_neuron_tag(config):
    if config.head:
        method_tag = f'head{config.head}'
    else:
        method_tag = re.sub('\.', 'p', f'thresh{config.thresh:.02f}')

    rwd = config.response_window
    param_tag = f'response_window{rwd[0]:d}to{rwd[1]:d}'

    if config.normalize_by_std:
        swd = config.std_window
        sdt_tag = f'std_window{swd[0]:d}to{swd[1]:d}'
        param_tag = param_tag + '_' + std_tag

    tag = method_tag + '_' + param_tag
    return tag


def _get_top_n_positions(df, n):
    # Identify the positions of the top n values for each row
    top_positions_per_row = df.apply(lambda x: list(x.argsort()[::-1].head(n).values), axis=1)
    all_positions = [position for positions in top_positions_per_row for position in positions]
    return all_positions


def _sample_positions(all_positions, sample_size=50):
    # Randomly select sample_size positions from the list
    unique_positions = list(set(all_positions))
    if len(unique_positions) <= sample_size:
        raise ValueError(f'Cannot sample {sample_size} neurons from total {len(unique_positions)} neurons.')
    return list(np.random.choice(unique_positions, size=sample_size, replace=False))


def select_neuron_by_ensemble(dff, window, top_n_per_odor):
    dff = select_time_points(dff, window)
    response = dff.groupby(level='odor').mean().max()
    all_positions = _get_top_n_positions(dff, top_n_per_odor)
    # pos = _sample_positions(all_positions, sample_size)
    idx = dff.columns[all_positions]
    return idx


def get_select_neuron_func(criterion_func, thresh=None, head=None):
    def func(dfovf, **kwargs):
        criteria = criterion_func(dfovf, **kwargs)
        if thresh:
            idx = criteria >= thresh
        elif head:
            idx = criteria.index[:head]
        else:
            raise ValueError('To Choose a correct selection method, set either the parameter thresh or head')
        return idx
    return func


@dataclass_json
@dataclass
class SelectEnsembleConfig:
    window: list[str] # although named time window, by far it corresponds to the frame window
    top_n_per_odor: int

def sample_neuron(dfovf, sample_size):
    return dfovf.sample(n=sample_size, axis=1)


def select_neuron(dfovf, select_func, **kwargs):
    idx = select_func(dfovf, **kwargs)
    dfovf_select = dfovf.loc[:,idx]
    return dfovf_select, idx


def select_neuron_df(dfovf, **kwargs):
    dfovf_select, _ = select_neuron(dfovf, **kwargs)
    return dfovf_select


def select_neuron_dfovf(dfovf, **kwargs):
    """
    Select neuron based on their response/noise ratio

    use this for raw dfovf as input
    """
    dfovf = restack_as_pattern(dfovf)
    dfovf_select, _ = select_neuron(dfovf, **kwargs)
    return dfovf_select


def select_neuron_and_sort_odors(df, odor_list, **kwargs):
    df = select_neuron_df(df, **kwargs)
    df = sort_odors(df, odor_list)
    return df


def select_neurons_by_df(source_df, select_df):
    """
    Select in source_df the columns that also appears in select_df
    """
    return source_df.loc[select_df.index]


def average_trials(tracedf):
    pass


def permute_odors(df):
    odor_list = df.index.unique('odor').tolist()
    random.shuffle(odor_list)
    return df.reindex(odor_list, level='odor')


def select_odors_df(df, odors):
    sedf = df.loc[df.index.get_level_values('odor').isin(odors)].copy()
    odor_idxs = sedf.index.get_level_values('odor')
    if not isinstance(odor_idxs.dtype, pd.CategoricalDtype):
        odor_idxs = pd.Categorical(odor_idxs)
    odor_idxs = odor_idxs.remove_unused_categories()
    sedf.loc[:, 'new_odor'] = odor_idxs
    sedf.set_index('new_odor', append=True, inplace=True)
    sedf.index = sedf.index.droplevel('odor')
    sedf.rename_axis(index={'new_odor': 'odor'}, inplace=True)
    return sedf


def sort_odors(df, odor_list):
    cat_odor = pd.CategoricalDtype(categories=odor_list, ordered=True)
    idx = df.index.names.index('odor')
    df.index = df.index.set_levels(df.index.levels[idx].astype(cat_odor),
                             level='odor')
    # df = df.sort_values('odor') # this alters the order of other levels
    groups = []
    keys = []
    for key, group in df.groupby(level='odor'):
        groups.append(group)
        keys.append(key)
    df_sorted = pd.concat(groups)
    return df_sorted


def select_time_points(df, window):
    """Select trace dataframe in a given time window"""
    if 'time' in df.index.names:
        df_filtered = df[(df.index.get_level_values('time') >= window[0])
                        & (df.index.get_level_values('time') <= window[1])]
    elif 'time_bin' in df.index.names:
        df_filtered = select_binned_time_points(df, window)
    else:
        raise ValueError('Dataframe index names should contain time or time_bin')

    return df_filtered


def select_binned_time_points(df, window):
    interval = pd.Interval(left=window[0], right=window[1])
    selected_index = [tb.overlaps(interval) for tb in df.index.get_level_values('time_bin')]
    df_sel = df[selected_index]
    return df_sel


@dataclass_json
@dataclass
class SelectDfConfig:
    odors: list[str]
    time_window: list[str] # although named time window, by far it corresponds to the frame window


def select_dataframe(df: pd.DataFrame, config: SelectDfConfig):
    df = select_time_points(df, config.time_window)
    df = select_odors_df(df, config.odors)
    return df


def get_select_tag(config: SelectDfConfig):
    odor_tag = ''.join([x[0] for x in config.odors])
    window = config.time_window
    tag = f'odors_{odor_tag}_window{window[0]}to{window[1]}'
    return tag
