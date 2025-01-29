import os
import sys
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from importlib import reload
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .dataio import load_trace_file
from .process_time_trace import mean_pattern_in_time_window
from .process_neuron import sample_neuron
from .mahal import compute_distances_mat, compute_center_euclidean_distance_mat

def cosine_distance(mat):
    # Compute the pairwise cosine distances between trials
    distances = pdist(mat, metric='cosine')
    # Convert the condensed distance matrix to a square matrix
    sim_mat = squareform(distances)
    return sim_mat


def pattern_correlation(mat):
    # Compute the pairwise correlation between trials
    corr_mat = np.corrcoef(mat)
    return corr_mat


import numpy as np
from scipy.spatial.distance import cdist

def cosine_distance_to_template(mat, template):
    """
    Compute the cosine distances between each row in `mat` and the `template` vector.

    Parameters:
    mat (numpy.ndarray): A 2D array of shape (n_trials, n_features), where each row is a trial.
    template (numpy.ndarray): A 1D array of shape (n_features,), representing the template vector.

    Returns:
    numpy.ndarray: A 1D array of shape (n_trials,) containing the cosine distances.
    """
    # Ensure the template is a 2D array with shape (1, n_features)
    template_2d = template.to_numpy().reshape(1, -1)
    # Compute cosine distances between each row of mat and the template
    distances = cdist(mat, template_2d, metric='cosine')
    # Flatten the distances to a 1D array
    return distances.flatten()


def pattern_correlation_to_template(mat, template):
    """
    Compute the Pearson correlation between each row in `mat` and the `template` vector using np.corrcoef.

    Parameters:
    mat (numpy.ndarray): A 2D array of shape (n_trials, n_features), where each row is a trial.
        Each row represents an observation.
    template (numpy.ndarray): A 1D array of shape (n_features,), representing the template vector.

    Returns:
    numpy.ndarray: A 1D array of shape (n_trials,) containing the correlation coefficients.
    """
    # Check shapes of mat and template
    assert mat.shape[1] == template.shape[0], "The number of features in mat and template must match."
    n_trials = mat.shape[0]
    correlations = np.empty(n_trials)
    for i in range(n_trials):
        # Compute the correlation matrix between the ith row and the template
        corr_matrix = np.corrcoef(mat[i], template)
        # Extract the correlation coefficient
        correlations[i] = corr_matrix[0, 1]
    return correlations


def compute_similarity_mat(dfovf, time_window, frame_rate, similarity_func):
    """
    Compute of similarity matrix from response patterns of neurons
        Args:
            dfovf
            time_window
            frame_rate
            similarity_func: np.corrcoef or scipy.spatial.distance.cosine
    """
    pattern = mean_pattern_in_time_window(dfovf, time_window, frame_rate)
    pattern_mat = pattern.to_numpy()
    sim_mat = similarity_func(pattern_mat)
    sim_mat = pd.DataFrame(sim_mat, index=pattern.index, columns=pattern.index)
    return sim_mat


def compute_similarity_mat_timecourse(dff, window_size, similarity_func, window_method='sliding'):
    """Compute similarity matrix timecourse."""
    times = dff.index.unique(level='time')
    start_time = times[0]
    end_time = times[-1]

    if window_method == 'piecewise':
        # Generate windows using np.arange
        windows = [(start, min(start + window_size, end_time)) 
                for start in np.arange(start_time, end_time, window_size)]
    elif window_method == 'sliding':
        windows = [(start, start + window_size) for start in times[:-window_size]]
    else:
        raise ValueError('Invalid window_method. Choose from "piecewise" or "sliding".')        

    corrmat_tvec = []
    for window in windows:
        corrmat = compute_similarity_mat(dff, window, frame_rate=1, similarity_func=similarity_func)
        corrmat_tvec.append(corrmat)
    
    # Combine the list of correlation matrices into a single DataFrame
    corrmat_df = pd.concat(corrmat_tvec, keys=windows, names=['start_time', 'end_time'])
    
    return corrmat_df


def plot_correlation_timecourse(corrmat_df, row_col_indices, ax=None, color='blue', label='Mean'):
    """
    Plot the mean and standard deviation of correlation time traces across time.
    
    Parameters:
    - corrmat_df: DataFrame with correlation matrices, indexed by (start_time, end_time, odor, trial).
    - row_col_indices: List of tuples, each containing:
        - First element: tuple (odor, trial) for row selection.
        - Second element: tuple (odor, trial) for column selection.
    """
    
    time_traces = []
    
    # Extract the time traces based on the provided row_col_indices
    for (row_idx, col_idx) in row_col_indices:
        trace = corrmat_df.xs(row_idx, level=['odor', 'trial'], axis=0).xs(col_idx, level=['odor', 'trial'], axis=1)
        time_traces.append(trace)
    
    # Convert list of time traces into a DataFrame
    time_traces_df = pd.concat(time_traces, axis=1)
    
    # Calculate mean and standard deviation across time
    mean_trace = time_traces_df.mean(axis=1)
    std_trace = time_traces_df.std(axis=1)
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean trace
    sns.lineplot(x=mean_trace.index.get_level_values('start_time'), y=mean_trace, label=label, ax=ax, color=color)
    
    # Plot standard deviation as shaded area
    ax.fill_between(mean_trace.index.get_level_values('start_time'), 
                    mean_trace - std_trace, 
                    mean_trace + std_trace, 
                    color=color, alpha=0.3)#, label='Std Dev')
    
    # Set plot labels and title
    ax.set_title('Mean and Std Dev of Correlation Timecourse')
    ax.set_xlabel('Time')
    ax.set_ylabel('Correlation')
    ax.legend()


def get_same_odor_diff_trial(corrmat_df):
    """
    Generate row_col_indices for upper right corners of sub-matrices in corrmat_df.
    These correspond to the entries between different trials but the same odors.
    
    Parameters:
    - corrmat_df: DataFrame with correlation matrices, indexed by (start_time, end_time, odor, trial).
    
    Returns:
    - row_col_indices: List of tuples where each tuple contains:
        - First element: tuple (odor, trial) for row selection.
        - Second element: tuple (odor, trial) for column selection.
    """
    
    row_col_indices = []
    odors = corrmat_df.index.get_level_values('odor').unique()

    for odor in odors:
        trials = corrmat_df.loc[(slice(None), slice(None), odor, slice(None))].index.get_level_values('trial').unique()
        trial_pairs = [(trials[i], trials[j]) for i in range(len(trials)) for j in range(i + 1, len(trials))]
        
        for trial1, trial2 in trial_pairs:
            row_col_indices.append(((odor, trial1), (odor, trial2)))

    return row_col_indices

def plot_same_odor_diff_trial(corrmat_df, **kwargs):
    # Get the row_col_indices for same odor but different trials
    row_col_indices = get_same_odor_diff_trial(corrmat_df)
    
    # Plot the correlation timecourse using the calculated indices
    plot_correlation_timecourse(corrmat_df, row_col_indices, **kwargs)

from matplotlib.colors import Normalize

def plot_similarity_mat(df, ax=None, clim=None, cmap='RdBu_r', ylabel_fontsize=8, ylabel_colors=None, ylabels=None, title='', color_norm: Normalize = None):
    """
    Plot similarity matrix heatmap

    Args:
        **df**: pandas.DataFrame. Square matrix of pattern correlation.
        Row index levels: odor, trial. Column index levels: odor, trial.
        **ax**: plot Axis object. Axis to plot the matrix heatmap.
        **clim**: List. Color limit of the heatmap. Default ``None``.
        **title**: str. Title of the plot. Default ``''``.

    Returns:
        Image object.
    """
    im = ax.imshow(df.to_numpy(), cmap=cmap, norm=color_norm)

    if ylabel_colors is None:
        ylabel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    if ylabels is None:
        ylabels = [label for label in df.index.get_level_values('odor')]

    ylabel_unique = df.index.unique('odor')
    color_dict = dict(zip(ylabel_unique, ylabel_colors[:len(ylabels)]))
    tick_pos = np.arange(df.shape[0])
    ax.yaxis.set_tick_params(length=0)
    ax.set_yticks(tick_pos)
    print(f'ylabel_ vfontsize: {ylabel_fontsize}')
    ax.set_yticklabels(ylabels, fontsize=ylabel_fontsize)
    ax.set_xticks([])

    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(color_dict[ytick.get_text()])

    if clim:
        im.set_clim(clim)
    if title:
        ax.set_title(title)
    return im



def select_odors_mat(matdf, odors):
    if matdf.index.names == ['odor', 'trial']:
        smat = matdf.loc[(odors, slice(None)), (odors, slice(None))]
    else:
        smat = matdf.loc[odors, odors]
    return smat


def compute_aavsba(simdf, aa_odors, ba_odors):
    if simdf.index.names == ['odor', 'trial']:
        aavsba = simdf.loc[(aa_odors, slice(None)), (ba_odors, slice(None))].mean().mean()
    else:
        aavsba = simdf.loc[aa_odors, ba_odors].mean().mean()

    return aavsba


def sample_neuron_and_comopute_distance_mat(df, sample_size, seed=None, metric='center_euclidean',
                                            params={}, shuffle_seed_value=None):
    df = sample_neuron(df, sample_size=sample_size, seed=seed)
    params['shuffle_seed_value'] = shuffle_seed_value
    if metric in ['euclidean', 'mahal']:
        dist_mat = compute_distances_mat(df, **params)
    elif metric == 'center_euclidean':
        dist_mat = compute_center_euclidean_distance_mat(df, **params)
    else:
        raise ValueError('Invalid metric. Choose from "euclidean", "mahal", "center_euclidean".')
    return dist_mat


def extract_upper_triangle_similarities(simdf):
    """
    Extracts the upper triangle (excluding the diagonal) of similarity matrices
    grouped by the 'odor' level from a DataFrame with a MultiIndex.
    
    Parameters:
    simdf (pd.DataFrame): A DataFrame containing similarity matrices with a MultiIndex 
                          that has levels 'odor' and 'trial' in both the index and columns.
    
    Returns:
    pd.DataFrame: A DataFrame containing the 'odor' and the corresponding similarity values 
                  from the upper triangle of the matrices.
    """
    
    # Initialize an empty list to store the results
    results = []

    # Iterate over each group by the 'odor' level in the index
    for name, group in simdf.groupby(level='odor', observed=True):
        # Get the columns that odor is name
        group = group.xs(name, level='odor', axis=1, drop_level=False)
        # Extract the values of the similarity matrix
        values = group.values
        
        # Get the upper triangle of the matrix, excluding the diagonal
        upper_triangle = values[np.triu_indices(values.shape[0], k=1)]
        
        # Create a DataFrame with the results and add the 'odor' name
        upper_triangle_df = pd.DataFrame({
            'odor': name,
            'similarity_values': upper_triangle
        })
        
        upper_triangle_df.set_index('odor', inplace=True)
        # Append the DataFrame to the results list
        results.append(upper_triangle_df.T)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(results, axis=1)
    
    return final_df


# Statisitcs on CS odors
def reorder_cs(df, odor_to_cs, odor_orders):
    df = df.rename(index=odor_to_cs, level='odor')
    if 'ref_odor' in df.columns.names:
        df = df.rename(columns=odor_to_cs, level='ref_odor')
    else:
        df = df.rename(columns=odor_to_cs, level='odor')
    df = df.loc[odor_orders, odor_orders]
    return df


# Compute the difference to naive
def compute_diff_to_naive(simdf_list, exp_cond_list, do_reorder_cs=False, odor_orders=None, odors_aa=None, naive_name='naive'):
    if do_reorder_cs:
        if odor_orders is None or odors_aa is None:
            raise ValueError('For reordering CS, odor_orders and odors_aa must be provided.')
    
    naive_mats = [simdf for simdf, cond in zip(simdf_list, exp_cond_list) if cond == naive_name]
    mean_naive_mat = sum(naive_mats)  / len(naive_mats)

    trained_delta_mats = []
    for simdf, cond in zip(simdf_list, exp_cond_list):
        if cond == 'naive':
            continue
        else:
            if do_reorder_cs:
                cs_plus, cs_minus = cond.split('-')
                cs_plus = cs_plus.capitalize()
                cs_minus = cs_minus.capitalize()
                aa3 = [odor for odor in odors_aa if odor not in [cs_plus, cs_minus]][0]
                odor_to_cs = {cs_plus: 'cs_plus', cs_minus: 'cs_minus', aa3: 'aa3'}
                # Permute such that the amino acids order is CS+, CS-, AA3
                newdf = reorder_cs(simdf, odor_to_cs, odor_orders)
                new_naive_mat = reorder_cs(mean_naive_mat, odor_to_cs, odor_orders)
            else:
                newdf = simdf
                new_naive_mat = mean_naive_mat

            trained_delta_mats.append(newdf - new_naive_mat)
    mean_delta_mat = sum(trained_delta_mats) / len(trained_delta_mats)
    return mean_delta_mat    

@dataclass_json
@dataclass
class PlotMeanDeltaMatParams:
    figsize: tuple = (4, 4)
    colorbar_fontsize: float = 7
    ylabel_fontsize: float = 7
    ylabels: list = None
    ylabel_colors: list = None
    cmap: str = 'coolwarm'


from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_mean_delta_mat(mean_delta_mat: pd.DataFrame,
                        params: PlotMeanDeltaMatParams = PlotMeanDeltaMatParams()):
    cmin = mean_delta_mat.min().min()
    cmax = mean_delta_mat.max().max()
    print(cmin, cmax)
    abs_max = max(abs(cmin), abs(cmax))
    clim = (-abs_max, abs_max)

    params_dict = params.to_dict()
    figsize = params_dict.pop('figsize')
    fig, ax = plt.subplots(figsize=figsize)
    colorbar_fontsize = params_dict.pop('colorbar_fontsize')
    img = plot_similarity_mat(mean_delta_mat, ax=ax, clim=clim,
                              **params_dict)
    # Use make_axes_locatable to adjust the size of the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 'size' controls width, 'pad' controls spacing
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    # color bar tick only labels the min and max and zero
    cbar.set_ticks([clim[0], 0, clim[1]])
    #cbar = fig.colorbar(img, ax=ax)
    # cbar.ax.tick_params(labelsize=colorbar_fontsize)
    fig.tight_layout()
    return fig, ax


def average_mat_over_trials(simdf):
    #simdf has index levels: odor, trial
    #simdf has column levels: odor, trial
    # take average over trials
    avg_mat = simdf.groupby(level='odor', observed=True).mean().T.groupby(level='odor', observed=True).mean().T
    return avg_mat