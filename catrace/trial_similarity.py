

def plot_trial_similarity_mat(df, ax=None, clim=None, cmap='RdBu_r', ylabel_fontsize=7, color_norm: Normalize = None, frame_rate=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    im = ax.imshow(df.to_numpy(), cmap=cmap, norm=color_norm, interpolation='none')

    if frame_rate is not None:
        # Convert frame to time in seconds
        # Convert x axis from frame to time
        xticks = np.arange(0, df.shape[1], 1 * frame_rate)
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        xticklabels = xticks / frame_rate
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(xticklabels)
    return im


# def compute_similarity_over_time(trial_traces, bin_size=None, frame_range=None, similarity_method='pattern_correlation'):
#     """
#     Computes the similarity matrix and mean spike probability over time for the given trial traces.

#     Parameters:
#     - trial_traces: DataFrame, the trial traces data
#     - bin_size: int, the size of the bins for averaging (default is None)
#     - frame_range: tuple, the range of frames to consider (default is (18, 110))
#     - similarity_method: str, the similarity method to use, either 'cosine' or 'pattern_correlation'

#     Returns:
#     - xvec: The x-axis vector
#     - trial_traces_mean: Series, the mean spike probability over time
#     - mat: array, the computed similarity matrix
#     """
#     if frame_range is None:
#         frame_range = (0, len(trial_traces))
#     # Bin the trial traces if bin_size is specified
#     if bin_size is not None:
#         trial_traces = trial_traces.groupby(trial_traces.index // bin_size).mean()
#         frame_range = np.array(frame_range) // bin_size

#     # Adjust the frame range according to the bin size
#     trial_traces = trial_traces.iloc[frame_range[0]:frame_range[1], :]

#     # Define x-axis vector
#     xvec = np.arange(frame_range[0], frame_range[1]) - frame_range[0]

#     # Compute mean spike probability over cells (for plotting)
#     trial_traces_mean = trial_traces.mean(axis=1)

#     # Compute the similarity matrix based on the chosen method
#     if similarity_method == 'cosine':
#         mat = cosine_distance(trial_traces.to_numpy())
#     elif similarity_method == 'pattern_correlation':
#         mat = pattern_correlation(trial_traces.to_numpy())
#     else:
#         raise ValueError("Invalid similarity_method. Choose 'cosine' or 'pattern_correlation'.")

#     return xvec, trial_traces_mean, mat


def compute_trial_similarity_over_time(trial_traces, bin_size=None, similarity_method='pattern_correlation'):
    # Bin the trial traces if bin_size is specified
    if bin_size is not None:
        trial_traces = trial_traces.groupby(trial_traces.index // bin_size).mean()
        frame_range = np.array(frame_range) // bin_size

    # Compute the similarity matrix based on the chosen method
    if similarity_method == 'cosine':
        trial_similarity = cosine_distance(trial_traces.to_numpy())
    elif similarity_method == 'pattern_correlation':
        trial_similarity = pattern_correlation(trial_traces.to_numpy())
    else:
        raise ValueError("Invalid similarity_method. Choose 'cosine' or 'pattern_correlation'.")

    return trial_similarity


from matplotlib.colors import PowerNorm


def plot_correlation_over_time_subplots(xvec, trial_traces_mean, mat, similarity_method='pattern_correlation', clim=(0, 1),
                                        cmap='magma', power_norm=1.2, figsize=(5, 5)):
    """
    Plots the correlation over time with a line plot above a similarity matrix plot.
    Both plots share the same x-axis and have the same widths.

    Parameters:
    - xvec: array, the x-axis vector
    - trial_traces_mean: Series, the mean spike probability over time
    - mat: array, the similarity matrix
    - similarity_method: str, the similarity method used, either 'cosine' or 'pattern_correlation' (for labeling)
    - clim: tuple, the color limits for the similarity matrix (default is (0, 1))
    - cmap: str, the colormap for the similarity matrix (default is 'magma')
    - power_norm: float, the power normalization factor for the similarity matrix (default is 1.2)
    - figsize: tuple, the figure size (default is (5, 5))

    Returns:
    - fig, (ax_line, ax_mat): The figure and axes objects of the plot
    """
    # Create subplots with shared x-axis
    fig, (ax_line, ax_mat) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                          gridspec_kw={'height_ratios': [1, 4]}, constrained_layout=True)

    # Line plot on the top
    ax_line.plot(xvec, trial_traces_mean)
    ax_line.set_ylabel('Mean spike probability', fontsize=7)
    ax_line.tick_params(labelbottom=False)

    # Matshow plot below
    img = ax_mat.imshow(mat, interpolation='none', cmap=cmap, norm=PowerNorm(power_norm),
                        clim=clim, aspect='auto')

    # Set labels
    ax_mat.set_xlabel('Frames')
    ax_mat.set_ylabel('Frames')

    # Add colorbar without affecting the width
    cbar = fig.colorbar(img, ax=ax_mat, pad=0.02, fraction=0.046)
    clabel = 'Cosine Distance' if similarity_method == 'cosine' else 'Pattern Correlation'
    cbar.set_label(clabel, fontsize=7)

    return fig, (ax_line, ax_mat)

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class SimToTemplateParams:
    bin_size: int = None
    similarity_method: str = 'pattern_correlation'
    template_frame_range: tuple = None


def compute_trial_similarity_over_time_to_template(trial_traces, bin_size=None, similarity_method='pattern_correlation',
                                                   template=None, template_frame_range=None):
    # Bin the trial traces if bin_size is specified
    if bin_size is not None:
        trial_traces = trial_traces.groupby(trial_traces.index // bin_size).mean()

    times = trial_traces.index.get_level_values('time')
    idx = (times >= template_frame_range[0]) & (times <= template_frame_range[1])
    template = trial_traces.loc[idx, :].mean()

    # Compute the similarity matrix based on the chosen method
    if similarity_method == 'cosine':
        trial_similarity = cosine_distance_to_template(trial_traces.to_numpy(), template)
    elif similarity_method == 'pattern_correlation':
        trial_similarity = pattern_correlation_to_template(trial_traces.to_numpy(), template)
    else:
        raise ValueError("Invalid similarity_method. Choose 'cosine' or 'pattern_correlation'.")

    time_idx = trial_traces.index
    # Convert trial_similarity to a DataFrame with the row and column index to be the time index
    trial_similarity = pd.DataFrame(trial_similarity, index=time_idx)
    return trial_similarity