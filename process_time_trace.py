import numpy as np


def get_time_trace_dfovf(time_trace, fzero_window, intensity_offset=0,
                      fzero_percent=0.5):
    # Calculate dF/F from raw time traces
    # time_trace: a NxM matrix containing N traces of length M
    time_trace_fg = time_trace - intensity_offset
    fzero = np.quantile(time_trace_fg[:, fzero_window[0]:fzero_window[1]],
                        fzero_percent, axis=1)
    time_trace_df = (time_trace_fg - fzero[:, None]) / fzero[:, None]
    return time_trace_df
