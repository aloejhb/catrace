import os
import numpy as np
from itertools import product
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pandas as pd
from ..dataset import load_dataset_config
from .run_distance import PlotDistanceParams
from ..exp_collection import process_data_db_parallel, read_df
from ..visualize import (
    PlotBoxplotParams,
    PlotPerCondMatParams,
    plot_measure,
    plot_conds_mat,
    move_pvalue_indicator,
    plot_measure_multi_odor_cond,
    PlotBoxplotMultiOdorCondParams,
)
from .run_utils import plot_avg_trace_with_window, get_group_vs_group
from ..lda import sample_and_compute_lda, LdaCrossValParameters


@dataclass_json
@dataclass
class ComputeLdaManifoldPairParams:
    in_dir: str
    exp_list: list
    lda_cross_val_params: LdaCrossValParameters
    time_window: np.ndarray
    seed: int
    num_repeats: int
    sample_size: int
    overwrite_computation: bool
    parallelism: int
    manifold_level: str
    manifold_names: list


def compute_lda_manifold_pair(params: ComputeLdaManifoldPairParams):
    in_dir = params.in_dir
    exp_list = params.exp_list
    metric = 'lda_cv_acc_manifold_pair'
    time_window = params.time_window
    seed = params.seed
    num_repeats = params.num_repeats
    sample_size = params.sample_size
    manifold_level = params.manifold_level

    out_dir_name = (
        f'{metric}_seed{seed}_window{time_window[0]}to{time_window[1]}'
    )
    out_dir = os.path.join(in_dir, out_dir_name)

    if not os.path.exists(out_dir) or params.overwrite_computation:
        os.makedirs(out_dir, exist_ok=True)
        num_exp = len(exp_list)
        master_rng = np.random.default_rng(seed)
        seeds = [
            master_rng.integers(0, 1e9, size=num_exp).tolist()
            for _ in range(num_repeats)
        ]

        for k in range(num_repeats):
            out_dir = os.path.join(out_dir, f'repeat{k:02d}')
            sample_and_dist_params = dict(
                sample_size=sample_size,
                ordered_manifold_labels=params.manifold_names,
                time_window=time_window,
                manifold_level=manifold_level,
            )
            sample_and_compute_params = dict(
                lda_params=params.lda_cross_val_params,
                manifold_names=params.manifold_names,
                time_window=time_window,
                sample_size=sample_size
            )
            process_data_db_parallel(
                sample_and_compute_lda,
                exp_list,
                out_dir,
                in_dir,
                parallelism=params.parallelism,
                seeds=seeds[k],
                params=sample_and_compute_params,
            )
    return out_dir


def read_dfs_from_dir(dist_dir, exp_list, num_repeats):
    # Multiply sample numbers with exp_list
    keys = list(product(exp_list, range(num_repeats)))
    keys = [(exp[0], exp[1], k) for exp, k in keys]

    simdf_lists = []
    for exp_name, condition, k in keys:
        in_dir = os.path.join(dist_dir, f'repeat{k:02d}')
        simdf = read_df(in_dir, exp_name, verbose=False)
        simdf_lists.append(simdf)
    all_simdf = pd.concat(
        simdf_lists, keys=keys, names=['fish_id', 'condition', 'sample']
    )
    return all_simdf


@dataclass_json
@dataclass
class RunLdaManifoldPairParams:
    config_file: str
    do_reorder_cs: bool
    compute_lda_params: ComputeLdaManifoldPairParams
    odor_orders: list = None
    naive_name: str = 'naive'
    overwrite_computation: bool = False
    vs_same_ylim: list = None
    do_compare_cs: bool = False
    vsdict: dict = None
    plot_params: PlotDistanceParams = None


def run_lda_manifold_pair(params: RunLdaManifoldPairParams):
    dsconfig = load_dataset_config(params.config_file)
    exp_list = dsconfig.exp_list
    trace_dir = dsconfig.processed_trace_dir
    time_window = np.array(params.compute_lda_params.time_window)
    in_dir = trace_dir

    print('Plotting average trace...')
    fig_avg_trace, ax = plot_avg_trace_with_window(
        in_dir, exp_list[0][0], time_window
    )

    print('Computing distance matrices...')
    lda_dir = compute_lda_manifold_pair(params.compute_lda_params)

    all_lda_df = read_dfs_from_dir(
        lda_dir, exp_list, params.compute_lda_params.num_repeats
    )
    avg_lda_df = average_over_repeats(all_lda_df)

    # Reshape the dataframes into matrix forms
    # TODO
    avg_simdf = reshape_pair_df_to_matrix(lda_df)

    print('Plotting per condition...')
    per_cond_params = params.plot_params.per_cond
    print(per_cond_params.to_dict())
    fig_per_cond, axs = plot_matrix_per_condition(
        avg_simdf, dsconfig.conditions, params=per_cond_params
    )

    print('Plotting delta matrix...')
    if params.do_reorder_cs:
        mean_delta_mat = compute_diff_to_naive_from_simdfdf(
            avg_simdf,
            params.do_reorder_cs,
            params.odor_orders,
            dsconfig.odors_aa,
            naive_name=params.naive_name,
        )
    else:
        mean_delta_mat = compute_diff_to_naive_from_simdfdf(
            avg_simdf,
            do_reorder_cs=params.do_reorder_cs,
            naive_name=params.naive_name,
        )
    fig_delta, ax = plot_mean_delta_mat(
        mean_delta_mat, params.plot_params.mean_delta
    )

    print('Plotting vs statistics...')
    if params.vsdict is None:
        vsdict = {
            'aa_vs_aa': (dsconfig.odors_aa, dsconfig.odors_aa),
            'aa_vs_ba': (dsconfig.odors_aa, dsconfig.odors_ba),
            'ba_vs_aa': (dsconfig.odors_ba, dsconfig.odors_aa),
            'ba_vs_ba': (dsconfig.odors_ba, dsconfig.odors_ba),
        }
    else:
        vsdict = params.vsdict

    subsimdfs = {}
    for vsname, (group1, group2) in vsdict.items():
        try:
            pooled_subsimdf = pool_odor_pair(
                group1,
                group2,
                dsconfig.conditions,
                avg_simdf,
                metric,
                naive_name=params.naive_name,
            )
        except Exception as err:
            print(f'Error in {vsname}')
            raise err
        subsimdfs[vsname] = pooled_subsimdf

    vsdff = pd.concat(
        subsimdfs.values(), keys=subsimdfs.keys(), names=['vsname']
    )
    fig_multi_vs, ax, test_results = plot_measure_multi_odor_cond(
        vsdff,
        measure_name,
        odor_name='vsname',
        condition_name='condition',
        params=params.plot_params.vs_measure,
    )

    output_figs = dict(
        fig_avg_trace=fig_avg_trace,
        fig_per_cond=fig_per_cond,
        fig_delta=fig_delta,
        fig_multi_vs=fig_multi_vs,
    )

    return output_figs, test_results, concat_subsimdf
