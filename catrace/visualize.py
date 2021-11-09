def plot_pattern_heatmap(pattern, climit, ax):
    im = ax.imshow(pattern.T, aspect='auto', interpolation='none',
                   vmin=climit[0], vmax=climit[1])
    return im
