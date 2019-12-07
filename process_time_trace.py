import numpy as np
import pandas as pd
from .frame_time import convert_sec_to_frame


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

