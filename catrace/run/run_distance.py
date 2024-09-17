import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import catrace.exp_collection as ecl
import catrace.pattern_correlation as catpcr
import catrace.mahal as catmah


from os.path import join as pjoin
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from scipy.stats import mannwhitneyu

from catrace.dataset import DatasetConfig
from catrace.utils import load_config
from catrace.stats import pool_training_conditions
from catrace.visualize import plot_measure, plot_conds_mat, move_pvalue_indicator
from catrace.similarity import plot_similarity_mat, sample_neuron_and_comopute_distance_mat


# dataclass for compute dist
@dataclass_json
@dataclass
class ComputeDistParams:
    in_dir: str
    exp_list: list
    metric: str
    time_window: list
    seed: int
    num_repeats: int
    sample_size: int
    reg: float
    odors: list
    overwrite_computation: bool
    parallelism: int


def plot_avg_trace_with_window(trace_dir, exp_name, window):
    dff = ecl.read_df(trace_dir, exp_name)
    avg_trace = dff.groupby(level='time').mean().mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(avg_trace.index.get_level_values('time'), avg_trace.to_numpy())
    ax.axvline(window[0], linestyle='--', color='red')
    ax.axvline(window[1], linestyle='--', color='red')
    return fig, ax


def compute_dist(params: ComputeDistParams):
    in_dir = params.in_dir
    exp_list = params.exp_list
    metric = params.metric
    time_window = params.time_window
    seed = params.seed
    num_repeats = params.num_repeats
    sample_size = params.sample_size
    reg = params.reg
    odors = params.odors
    overwrite_computation = params.overwrite_computation
    parallelism = params.parallelism

    dist_dir = pjoin(in_dir, f'{metric}_seed{seed}_window{time_window[0]}to{time_window[1]}')

    if not os.path.exists(dist_dir) or overwrite_computation:
        os.makedirs(dist_dir, exist_ok=True)
        num_exp = len(exp_list)
        master_rng = np.random.default_rng(seed)
        seeds = [
            master_rng.integers(0, 1e9, size=num_exp).tolist() 
            for _ in range(num_repeats)
        ]

        for k in range(num_repeats):
            out_dir = pjoin(dist_dir, f'repeat{k:02d}')
            dist_params=dict(odor_list=odors, window=time_window)
            if metric in ['mahal', 'euclidean']:
                dist_params.update(dict(metric=metric, reg=reg))
            sample_and_dist_params = dict(sample_size=sample_size,
                                          metric=params.metric,
                                          params=dist_params)
            ecl.process_data_db_parallel(sample_neuron_and_comopute_distance_mat, exp_list,
                                        out_dir, in_dir, parallelism=parallelism, seeds=seeds[k],
                                        params=sample_and_dist_params)
    return dist_dir

# Read the matrices
def read_mats_from_dir(dist_dir, exp_list, num_repeats):
    # Multiply sample numbers with exp_list
    keys = list(product(exp_list, range(num_repeats)))
    keys = [(exp[0], exp[1], k) for exp, k in keys]

    simdf_lists = []
    for exp_name, cond, k in keys:
        in_dir = pjoin(dist_dir, f'repeat{k:02d}')
        simdf = ecl.read_df(in_dir, exp_name, verbose=False)
        simdf.index.name = 'odor'
        simdf_lists.append(simdf)
    all_simdf = pd.concat(simdf_lists, keys=keys, names=['fish_id', 'condition', 'sample'])
    return all_simdf


# Average over repeats
def average_over_repeats(all_simdf):
    avg_simdf = all_simdf.groupby(['fish_id', 'condition', 'odor'], observed=True, sort=False).mean()
    return avg_simdf


# For each fish_id, get the average similarity matrix, normalize it by the sum of all entries
def normalize_simdf(avg_simdf):
    for (fish_id, cond), simdf in avg_simdf.groupby(['fish_id', 'condition']):
        simdf = simdf / simdf.sum().sum()
        avg_simdf.loc[(fish_id, cond), :] = simdf
    return avg_simdf


def get_mat_lists(matdf, fish_ids=None):
    # group by fish_id and condition and drop the fish_id and condition levels, and sort the groups by fish_id according to the order of fish_ids
    grouped = matdf.groupby(['fish_id', 'condition'])
    if fish_ids is not None:
        grouped = sorted(grouped, key=lambda x: fish_ids.index(x[0][0]))
    data_list = [group.droplevel(['fish_id', 'condition']) for name, group in grouped]
    exp_cond_list = [name[1] for name, group in grouped]
    return data_list, exp_cond_list


# Plot matrix per fish
def plot_matrix_per_fish(avg_simdf, cmap='turbo'):
    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)

    for simdf in simdf_list:
        simdf.index.name = 'odor'

    cmin = min([mat.min().min() for mat in simdf_list])
    cmax = max([mat.max().max() for mat in simdf_list])
    fig, axs = ecl.plot_explist_with_cond(simdf_list, exp_cond_list, plot_similarity_mat, clim=(cmin, cmax), cmap=cmap)
    ax = axs[-1, 0]
    img = ax.get_children()[0]
    fig.colorbar(img, ax=ax)
    return fig, axs


# Plot matrix per condition
def plot_matrix_per_condition(avg_simdf, conditions, cmap='turbo', clim=None):
    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)
    avg_mats = ecl.mean_mat_over_cond(simdf_list, exp_cond_list, conditions)
    if clim is None:
        cmin = min([mat.min().min() for mat in avg_mats.values()])
        cmax = max([mat.max().max() for mat in avg_mats.values()])
        clim = (cmin, cmax)
    fig, axs = plot_conds_mat(avg_mats, conditions, plot_similarity_mat, clim=clim, cmap=cmap, ncol=2, ylabel_fontsize=12)
    return fig, axs


# def get_group_vs_group(avg_simdf, odor_group1, odor_group2):
#     if 'sample' in avg_simdf.index.names:
#         avg_simdf = avg_simdf.loc[(slice(None), slice(None), slice(None), odor_group1), odor_group2]
#     else:
#         avg_simdf = avg_simdf.loc[(slice(None), slice(None), odor_group1), odor_group2]
#     return avg_simdf


import pandas as pd

def get_group_vs_group(dff, odor1_group, odor2_group, measure_name, deduplicate=True):
    # Get tuples of (odor1, odor2) that are in the group
    odor_tuples = [(odor1, odor2) for odor1 in odor1_group for odor2 in odor2_group]
    # Remove the tuples that have the same odor1 and odor2
    odor_tuples = [odor_tuple for odor_tuple in odor_tuples if odor_tuple[0] != odor_tuple[1]]
    
    # Conditionally deduplicate the odor_tuples by sorting and removing duplicates
    if deduplicate:
        odor_tuples = list({tuple(sorted(odor_tuple)) for odor_tuple in odor_tuples})
    
    # Stack the DataFrame so that 'ref_odor' becomes part of the MultiIndex
    dff_stacked = dff.stack()  # This moves the columns (ref_odor) into the index
    
    # Rename the stacked column for clarity
    dff_stacked.name = 'value'
    
    # Use pd.IndexSlice to slice through the MultiIndex
    idx = pd.IndexSlice
    
    # Prepare the filter based on odor_tuples
    filtered_dfs = []
    for odor1, odor2 in odor_tuples:
        # Slice through MultiIndex using `odor1` and `odor2` while preserving other levels
        if 'sample' in dff_stacked.index.names:  # Adjust for additional levels (like 'sample')
            sliced_df = dff_stacked.loc[idx[:, :, :, odor1, odor2]]
        else:  # Default case with two extra levels before 'odor' and 'ref_odor'
            sliced_df = dff_stacked.loc[idx[:, :, odor1, odor2]]
        
        filtered_dfs.append(sliced_df)
    
    # Concatenate all the filtered DataFrames
    gvg = pd.concat(filtered_dfs, keys=odor_tuples)
    
    # Rename the first two levels of the index to 'odor' and 'ref_odor'
    new_index_names = list(gvg.index.names)
    
    if len(new_index_names) >= 2:  # Ensure there are at least two levels to rename
        new_index_names[0] = 'odor'      # First level becomes 'odor'
        new_index_names[1] = 'ref_odor'  # Second level becomes 'ref_odor'
    
    # Apply the new index names
    gvg = gvg.rename_axis(new_index_names, axis=0)
    
    # Convert the final gvg to a DataFrame with the specified column name (measure_name)
    gvg_df = gvg.to_frame(name=measure_name)
    
    return gvg_df


# Statistics on odors
def stat_of_odor_pair(group1, group2, selected_conditions, avg_simdf, metric, vsname, naive_name='naive'):
    if metric == 'mahal':
        deduplicate = False
    else:
        deduplicate = True
    if metric == 'mahal':
        measure_name = 'D_M'
    elif metric == 'euclidean':
        measure_name = 'D_E'
    elif metric == 'center_euclidean':
        measure_name = 'center D_E'
    else:
        raise ValueError(f'Unknown metric: {metric}')
    subsimdf = get_group_vs_group(avg_simdf, group1, group2, measure_name=measure_name,
                                  deduplicate=deduplicate)
    # Select conditions
    subsimdf = subsimdf[subsimdf.index.get_level_values('condition').isin(selected_conditions)]
    # subsimdf = subsimdf.loc[(slice(None), selected_conditions, slice(None)), :]
    # subsimdf = subsimdf.stack('ref_odor').to_frame()
    # Remove the rows where odor is the same as ref_odor

    # rename the column as D_E or D_M

    # subsimdf.rename(columns={0: measure_name}, inplace=True)

    # Map naive to naive, others to trained
    condition_map = {cond: 'trained' if cond != naive_name else 'naive' for cond in selected_conditions}
    pooled_subsimdf = pool_training_conditions(subsimdf, condition_map)

    fig, ax, test_results = plot_measure(measure_name, pooled_subsimdf, test_type='mannwhitneyu')
    # Title
    fig.suptitle(vsname)
    # tight layout
    fig.tight_layout()
    return fig, ax, test_results, pooled_subsimdf


def normalize_to_percent(pooled_subsimdf, normalize=True):
    # From pooled_subsimdf, get the rows where condition is naive
    pooled_subsimdf_naive = pooled_subsimdf[pooled_subsimdf.index.get_level_values('condition') == 'naive']
    # Compute the mean of naive
    pooled_subsimdf_naive_mean = pooled_subsimdf_naive.groupby(['odor', 'ref_odor']).mean()
    # Subtract the pooled_subsimdf_naive_mean from pooled_subsimdf and then normalize by the pooled_subsimdf_naive_mean
    pooled_subsimdf_normailized = pooled_subsimdf - pooled_subsimdf_naive_mean
    if normalize:
        pooled_subsimdf_normailized = pooled_subsimdf_normailized / pooled_subsimdf_naive_mean * 100
    # Get the rows where condition is not naive
    pooled_subsimdf_normalized_trained = pooled_subsimdf_normailized[pooled_subsimdf_normailized.index.get_level_values('condition') != 'naive']
    return pooled_subsimdf_normalized_trained


# Statisitcs on CS odors
def reorder_cs(df, odor_to_cs, odor_orders):
    df = df.rename(index=odor_to_cs, level='odor')
    df = df.rename(columns=odor_to_cs, level='ref_odor')
    df = df.loc[odor_orders, odor_orders]
    return df


# Compute the difference to naive
def compute_diff_to_naive(avg_simdf, do_reorder_cs=False, odor_orders=None, odors_aa=None, naive_name='naive'):
    if do_reorder_cs:
        if odor_orders is None or odors_aa is None:
            raise ValueError('For reordering CS, odor_orders and odors_aa must be provided.')
    
    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)
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


def plot_mean_delta_mat(mean_delta_mat):
    cmin = mean_delta_mat.min().min()
    cmax = mean_delta_mat.max().max()
    print(cmin, cmax)
    abs_max = max(abs(cmin), abs(cmax))
    clim = (-abs_max, abs_max)

    fig, ax = plt.subplots()
    img = catpcr.plot_pattern_correlation(mean_delta_mat, ax=ax, cmap='coolwarm', clim=clim)
    fig.colorbar(img, ax=ax)
    return fig, ax


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def combine_figures_to_grid(figs, nrows, ncols):
    """
    Combines a list of figures as whole into a single figure with a grid of subfigures.

    Parameters:
        figs (list): List of matplotlib figures to combine.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.

    Returns:
        matplotlib.figure.Figure: The combined figure with a grid of subfigures.
    """
    if len(figs) > nrows * ncols:
        raise ValueError("The number of figures exceeds the grid capacity (nrows * ncols).")

    # Create the main figure
    fig = plt.figure(figsize=(ncols * 4, nrows * 4))

    # Create a grid of subfigures
    subfigs = fig.subfigures(nrows, ncols)
    subfigs = subfigs.flatten()  # Flatten for easier indexing

    # Loop through the provided figures and embed them in the subfigures
    for i, single_fig in enumerate(figs):
        if single_fig is None:
            continue
        # Draw the figure on a canvas to rasterize it
        canvas = FigureCanvas(single_fig)
        canvas.draw()

        # Get the rasterized image of the figure
        img = canvas.buffer_rgba()

        # Create a new axis in the subfigure and remove ticks/labels
        ax = subfigs[i].add_subplot(1, 1, 1)
        ax.imshow(img)
        ax.axis('off')  # Turn off axis labels and ticks to treat the figure as an image

    # Adjust layout
    fig.tight_layout()

    return fig


def compare_vs(vskeys, subsimdfs, measure_name):
    #vskeys = ['aa_vs_ba', 'aa_vs_aa']
    normalized_subsimdfs = {key: normalize_to_percent(subsimdfs[key]) for key in vskeys}
    concat_subsimdf = pd.concat([normalized_subsimdfs[vskeys[0]], normalized_subsimdfs[vskeys[1]]], keys=vskeys)
    # rename the level of keys to vsname
    concat_subsimdf.index = concat_subsimdf.index.rename('vsname', level=0)
    fig, ax = plt.subplots()
    sns.boxplot(x='vsname', y=measure_name, data=concat_subsimdf.reset_index())
    # Use mannwhitneyu to compare the two groups
    vs1 = normalized_subsimdfs[vskeys[0]][measure_name].to_numpy()
    vs2 = normalized_subsimdfs[vskeys[1]][measure_name].to_numpy()
    _, pvalue = mannwhitneyu(vs1, vs2)
    vs_tag = '__'.join(vskeys)
    ax.set_title(f'{vs_tag} p={pvalue:.6f}')
    return fig, ax, concat_subsimdf


@dataclass_json
@dataclass
class RunDistanceParams:
    config_file: str
    assembly_name: str
    time_window: list
    sample_size: int
    metric: str
    seed: int
    do_normalize_simdf: bool
    do_reorder_cs: bool
    overwrite_computation: bool
    report_dir: str
    summary_name: str
    save_output: bool
    odor_orders: list = None
    naive_name: str = 'naive'
    vs_same_ylim: list = None
    cmap: str = 'turbo'
    clim: list = None
    do_compare_cs: bool = False


def run_distance(params: RunDistanceParams):
    config_file = params.config_file
    assembly_name = params.assembly_name
    time_window = params.time_window
    sample_size = params.sample_size
    metric = params.metric
    seed = params.seed
    do_normalize_simdf = params.do_normalize_simdf
    overwrite_computation = params.overwrite_computation
    report_dir = params.report_dir
    summary_name = params.summary_name

    dsconfig= load_config(config_file, DatasetConfig)
    exp_list = dsconfig.exp_list
    trace_dir = dsconfig.processed_trace_dir
    select_neuron_dir = pjoin(trace_dir, assembly_name)
    time_window = np.array(time_window)
    in_dir = select_neuron_dir

    compute_dist_params = ComputeDistParams(in_dir=in_dir,
                                            exp_list=exp_list,
                                            metric=metric,
                                            time_window=time_window,
                                            seed=seed,
                                            num_repeats=50,
                                            sample_size=sample_size,
                                            reg=1e-5,
                                            odors=dsconfig.odors_stimuli, overwrite_computation=overwrite_computation,
                                            parallelism=16)

    print('Plotting average trace...')
    fig_avg_trace, ax = plot_avg_trace_with_window(trace_dir, exp_list[0][0], time_window)

    print('Computing distance matrices...')
    dist_dir = compute_dist(compute_dist_params)

    all_simdf = read_mats_from_dir(dist_dir, exp_list, compute_dist_params.num_repeats)
    avg_simdf = average_over_repeats(all_simdf)
    if do_normalize_simdf:
        avg_simdf = normalize_simdf(avg_simdf)
    
    print('Plotting per condition...')
    fig_per_cond, axs = plot_matrix_per_condition(avg_simdf, dsconfig.conditions, cmap=params.cmap, clim=params.clim)

    print('Plotting delta matrix...')
    if params.do_reorder_cs:
        mean_delta_mat = compute_diff_to_naive(avg_simdf, params.do_reorder_cs, params.odor_orders, dsconfig.odors_aa, naive_name=params.naive_name)
    else:
        mean_delta_mat = compute_diff_to_naive(avg_simdf, do_reorder_cs=params.do_reorder_cs, naive_name=params.naive_name)
    fig_delta, ax = plot_mean_delta_mat(mean_delta_mat)

    print('Plotting vs statistics...')
    vsdict = {'aa_vs_ba': (dsconfig.odors_aa, dsconfig.odors_ba),
              'aa_vs_aa': (dsconfig.odors_aa, dsconfig.odors_aa),
              'ba_vs_aa': (dsconfig.odors_ba, dsconfig.odors_aa)}
    if params.do_compare_cs:
        vsdict.update({'cs_vs_ba': (dsconfig.odors_cs, dsconfig.odors_ba),
                       'ba_vs_cs': (dsconfig.odors_ba, dsconfig.odors_cs),
                       'cs_plus_vs_cs_minus': ([dsconfig.odors_cs[0]], [dsconfig.odors_cs[1]])})
    vsfigs = []
    vsaxs = []
    stats = {}
    subsimdfs = {}
    for vsname, (group1, group2) in vsdict.items():
        fig, ax, test_results, pooled_subsimdf = stat_of_odor_pair(group1, group2, dsconfig.conditions, avg_simdf, metric, vsname, naive_name=params.naive_name)
        vsfigs.append(fig)
        vsaxs.append(ax)
        stats[vsname] = list(test_results.values())[0]
        subsimdfs[vsname] = pooled_subsimdf

    if metric == 'mahal':
        measure_name = 'D_M'
    elif metric == 'euclidean':
        measure_name = 'D_E'
    elif metric == 'center_euclidean':
        measure_name = 'center D_E'
    else:
        raise ValueError(f'Unknown metric: {metric}')
    # Compare percentage changes
    vskeys = ['aa_vs_ba', 'aa_vs_aa']
    fig, ax, concat_subsimdf = compare_vs(vskeys, subsimdfs, measure_name)
    if params.do_compare_cs:
        vskeys = ['cs_vs_ba', 'cs_plus_vs_cs_minus']
        fig, ax, _ = compare_vs(vskeys, subsimdfs, measure_name)


    if params.vs_same_ylim is not None:
        for vsax in vsaxs:
            vsax.set_ylim(params.vs_same_ylim)
            move_pvalue_indicator(vsax, params.vs_same_ylim[1])


    print('Plotting per fish...')
    fig_per_fish, axs = plot_matrix_per_fish(avg_simdf, cmap=params.cmap)

    if params.save_output:
        print('Saving stats...')
        # Save stats as json
        print(stats)
        stats_file = pjoin(report_dir, f'{summary_name}.json')
        with open(stats_file, 'w') as file:
            json.dump(stats, file)

        print('Saving summary...')
        with PdfPages(pjoin(report_dir, f'{summary_name}.pdf')) as pdf_pages:
            # Combine fig_per_cond, fig_delta vsfigs into one page
            figs = [fig_per_cond, fig_delta] + [None, fig_avg_trace] + vsfigs
            fig_combined = combine_figures_to_grid(figs, nrows=2, ncols=3)
            pdf_pages.savefig(fig_combined)
            pdf_pages.savefig(fig_per_fish)
    else:
        return fig_avg_trace, fig_per_cond, fig_delta, vsfigs, fig_per_fish, concat_subsimdf