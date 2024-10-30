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
from ..stats import pool_training_conditions
from ..similarity import (plot_similarity_mat,
                          sample_neuron_and_comopute_distance_mat,
                          compute_diff_to_naive,
                          plot_mean_delta_mat, PlotMeanDeltaMatParams)

from ..visualize import (PlotBoxplotParams, PlotPerCondMatParams,
                         plot_measure, plot_conds_mat, move_pvalue_indicator,
                         plot_measure_multi_odor_cond,
                         PlotBoxplotMultiOdorCondParams)
from .run_utils import plot_avg_trace_with_window


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
def plot_matrix_per_condition(avg_simdf, conditions,
                              params=PlotPerCondMatParams()):
    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)
    avg_mats = ecl.mean_mat_over_cond(simdf_list, exp_cond_list, conditions)
    if params.clim is None:
        cmin = min([mat.min().min() for mat in avg_mats.values()])
        cmax = max([mat.max().max() for mat in avg_mats.values()])
        clim = (cmin, cmax)
        params.clim = clim
    params.ncol = 2
    
    fig, axs = plot_conds_mat(avg_mats, conditions, plot_similarity_mat, **params.to_dict())
    return fig, axs


def get_group_vs_group(dff, odor1_group, odor2_group, measure_name, deduplicate=True):
    """
    Get the DataFrame containing the group vs. group data for the given DataFrame.

    Args:
        dff (pd.DataFrame): The DataFrame containing the data.
        The shape of dff is (n_samples, n_conditions, n_odors, n_odors).
        odor1_group (list): The list of odors in the first group.
        odor2_group (list): The list of odors in the second group.
        measure_name (str): The name of the measure column.
        deduplicate (bool): Whether to deduplicate the odor pairs.

    """
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
def pool_odor_pair(group1, group2, selected_conditions, avg_simdf, metric, naive_name='naive'):
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

    # Map naive to naive, others to trained
    condition_map = {cond: 'trained' if cond != naive_name else 'naive' for cond in selected_conditions}
    pooled_subsimdf = pool_training_conditions(subsimdf, condition_map)

    return pooled_subsimdf

def plot_single_vs(pooled_subsimdf, vsname, measure_name, title_fontsize=7, params=PlotBoxplotParams()):
    fig, ax, test_results = plot_measure(pooled_subsimdf, measure_name, test_type='mannwhitneyu', params=params)
    # Title
    fig.suptitle(vsname, fontsize=title_fontsize)
    # tight layout
    fig.tight_layout()


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


def compute_diff_to_naive_from_simdfdf(avg_simdf, *args, **kwargs):
    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)
    return compute_diff_to_naive(simdf_list, exp_cond_list, *args, **kwargs)


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
class PlotDistanceParams:
    per_cond: PlotPerCondMatParams
    mean_delta: PlotMeanDeltaMatParams
    vs_measure: PlotBoxplotMultiOdorCondParams = PlotBoxplotMultiOdorCondParams()


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
    odor_orders: list = None
    naive_name: str = 'naive'
    overwrite_computation: bool = False
    report_dir: str = None
    summary_name: str = None
    save_output: bool = False
    vs_same_ylim: list = None
    do_compare_cs: bool = False
    reg: float = 1e-5
    do_plot_per_fish: bool = False
    vsdict: dict = None
    plot_params: PlotDistanceParams = None




def run_distance(params: RunDistanceParams):
    assembly_name = params.assembly_name
    time_window = params.time_window
    sample_size = params.sample_size
    metric = params.metric
    seed = params.seed
    do_normalize_simdf = params.do_normalize_simdf
    overwrite_computation = params.overwrite_computation
    report_dir = params.report_dir
    summary_name = params.summary_name

    dsconfig= load_config(params.config_file, DatasetConfig)
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
                                            reg=params.reg,
                                            odors=dsconfig.odors_stimuli, overwrite_computation=overwrite_computation,
                                            parallelism=16)

    print('Plotting average trace...')
    fig_avg_trace, ax = plot_avg_trace_with_window(in_dir, exp_list[0][0], time_window)

    print('Computing distance matrices...')
    dist_dir = compute_dist(compute_dist_params)

    all_simdf = read_mats_from_dir(dist_dir, exp_list, compute_dist_params.num_repeats)
    avg_simdf = average_over_repeats(all_simdf)
    if do_normalize_simdf:
        avg_simdf = normalize_simdf(avg_simdf)
    
    print('Plotting per condition...')
    per_cond_params = params.plot_params.per_cond
    print(per_cond_params.to_dict())
    fig_per_cond, axs = plot_matrix_per_condition(avg_simdf, dsconfig.conditions, params=per_cond_params)

    print('Plotting delta matrix...')
    if params.do_reorder_cs:
        mean_delta_mat = compute_diff_to_naive_from_simdfdf(avg_simdf, params.do_reorder_cs, params.odor_orders, dsconfig.odors_aa, naive_name=params.naive_name)
    else:
        mean_delta_mat = compute_diff_to_naive_from_simdfdf(avg_simdf, do_reorder_cs=params.do_reorder_cs, naive_name=params.naive_name)
    fig_delta, ax = plot_mean_delta_mat(mean_delta_mat, params.plot_params.mean_delta)


    if metric == 'mahal':
        measure_name = 'D_M'
    elif metric == 'euclidean':
        measure_name = 'D_E'
    elif metric == 'center_euclidean':
        measure_name = 'center D_E'
    else:
        raise ValueError(f'Unknown metric: {metric}')



    print('Plotting vs statistics...')
    if params.vsdict is None:
        vsdict = {'aa_vs_aa': (dsconfig.odors_aa, dsconfig.odors_aa),
                'aa_vs_ba': (dsconfig.odors_aa, dsconfig.odors_ba),
                'ba_vs_aa': (dsconfig.odors_ba, dsconfig.odors_aa),
                'ba_vs_ba': (dsconfig.odors_ba, dsconfig.odors_ba)}
        if params.do_compare_cs:
            vsdict.update({'cs_vs_ba': (dsconfig.odors_cs, dsconfig.odors_ba),
                        'ba_vs_cs': (dsconfig.odors_ba, dsconfig.odors_cs),
                        'cs_plus_vs_cs_minus': ([dsconfig.odors_cs[0]], [dsconfig.odors_cs[1]])})
    else:
        vsdict = params.vsdict

    subsimdfs = {}
    for vsname, (group1, group2) in vsdict.items():
        try:
            pooled_subsimdf = pool_odor_pair(group1, group2, dsconfig.conditions, avg_simdf, metric, naive_name=params.naive_name)
        except Exception as err:
            print(f'Error in {vsname}')
            raise err
        subsimdfs[vsname] = pooled_subsimdf

    vsdff = pd.concat(subsimdfs.values(), keys=subsimdfs.keys(),
                      names=['vsname'])
    fig_multi_vs, ax, test_results = plot_measure_multi_odor_cond(vsdff, measure_name, odor_name='vsname', condition_name='condition', params=params.plot_params.vs_measure)
    

    
    if params.vsdict is None:
        # Compare percentage changes
        vskeys = ['aa_vs_aa', 'aa_vs_ba']
        fig, ax, concat_subsimdf = compare_vs(vskeys, subsimdfs, measure_name)
        vskeys = ['ba_vs_aa', 'aa_vs_ba']
        fig, ax, concat_subsimdf = compare_vs(vskeys, subsimdfs, measure_name)
        if params.do_compare_cs:
            vskeys = ['cs_vs_ba', 'cs_plus_vs_cs_minus']
            fig, ax, _ = compare_vs(vskeys, subsimdfs, measure_name)
    else:
        concat_subsimdf = None


    if params.vs_same_ylim is not None:
        #move_pvalue_indicator(vsax, params.vs_same_ylim[1])
        pass


    print('Plotting per fish...')
    if params.do_plot_per_fish:
        fig_per_fish, axs = plot_matrix_per_fish(avg_simdf, cmap=params.cmap)

    if params.save_output:
        print('Saving stats...')
        # Save stats as json
        # print(stats)
        # stats_file = pjoin(report_dir, f'{summary_name}.json')
        # with open(stats_file, 'w') as file:
        #     json.dump(stats, file)

        print('Saving summary...')
        with PdfPages(pjoin(report_dir, f'{summary_name}.pdf')) as pdf_pages:
            # Combine fig_per_cond, fig_delta vsfigs into one page
            figs = [fig_per_cond, fig_delta] + [None, fig_avg_trace]
            fig_combined = combine_figures_to_grid(figs, nrows=2, ncols=3)
            pdf_pages.savefig(fig_combined)
            if params.do_plot_per_fish:
                pdf_pages.savefig(fig_per_fish)

    output_figs = dict(
        fig_avg_trace=fig_avg_trace,
        fig_per_cond=fig_per_cond,
        fig_delta=fig_delta,
        fig_multi_vs=fig_multi_vs
    )
    if params.do_plot_per_fish:
        output_figs['fig_per_fish'] = fig_per_fish
    return output_figs, test_results, concat_subsimdf