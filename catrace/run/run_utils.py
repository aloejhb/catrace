import matplotlib.pyplot as plt
from ..exp_collection import read_df


def plot_avg_trace_with_window(trace_dir, exp_name, window):
    dff = read_df(trace_dir, exp_name)
    avg_trace = dff.groupby(level='time').mean().mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(avg_trace.index.get_level_values('time'), avg_trace.to_numpy())
    ax.axvline(window[0], linestyle='--', color='red')
    ax.axvline(window[1], linestyle='--', color='red')
    return fig, ax