import os
import numpy as np
from itertools import product
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pandas as pd
from ..dataset import load_dataset_config
from .run_distance import (PlotDistanceParams,
                           compute_diff_to_naive_from_simdfdf,
                           plot_mean_delta_mat,
                           plot_matrix_per_condition,
                           pool_odor_pair,
                           average_over_repeats)
from ..exp_collection import process_data_db_parallel, read_df
from ..visualize import plot_measure_multi_odor_cond
from .run_utils import plot_avg_trace_with_window, get_group_vs_group

from ..linear_separability import (  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    sample_and_compute_classifier,   # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    ClassifierCrossValParameters     # <<< MODIFIED (LDA -> LDA/SVM OPTION)
)                                   # <<< MODIFIED (LDA -> LDA/SVM OPTION)


@dataclass_json
@dataclass
class ComputeLinearSeparabilityManifoldPairParams:  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    in_dir: str
    exp_list: list
    cross_val_params: ClassifierCrossValParameters  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    time_window: np.ndarray
    seed: int
    num_repeats: int
    sample_size: int
    overwrite_computation: bool
    parallelism: int
    manifold_level: str
    manifold_names: list
    metric: str = 'cv_acc_manifold_pair'  # <<< MODIFIED (LDA -> LDA/SVM OPTION)


def compute_linear_separability_manifold_pair(params: ComputeLinearSeparabilityManifoldPairParams):  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    in_dir = params.in_dir
    exp_list = params.exp_list
    time_window = params.time_window
    seed = params.seed
    num_repeats = params.num_repeats
    sample_size = params.sample_size
    manifold_level = params.manifold_level
    params.cross_val_params.manifold_level = manifold_level  # <<< MODIFIED (LDA -> LDA/SVM OPTION)

    # Put model name into output directory + metric
    model_name = params.cross_val_params.model.lower()  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    metric_name = f'{model_name}_cv_acc_manifold_pair'  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    params.metric = metric_name  # <<< MODIFIED (LDA -> LDA/SVM OPTION)

    out_parent_dir_name = (  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
        f'{params.metric}_sample_size{sample_size}_seed{seed}_window{time_window[0]}to{time_window[1]}'  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    )

    # Model-specific labeling in dir name
    if model_name == 'lda':  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
        if params.cross_val_params.solver is not None:  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
            out_parent_dir_name += (  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
                f'_solver_{params.cross_val_params.solver}_shrinkage_{params.cross_val_params.shrinkage}'  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
            )
    elif model_name == 'svm':  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
        out_parent_dir_name += (  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
            f'_kernel_{params.cross_val_params.kernel}'
            f'_C_{params.cross_val_params.C}'
            f'_gamma_{params.cross_val_params.gamma}'
        )

    out_parent_dir_name += f'_k_{params.cross_val_params.k}'  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    out_parent_dir = os.path.join(in_dir, out_parent_dir_name)

    if not os.path.exists(out_parent_dir) or params.overwrite_computation:
        os.makedirs(out_parent_dir, exist_ok=True)
        num_exp = len(exp_list)
        master_rng = np.random.default_rng(seed)
        seeds = [
            master_rng.integers(0, 1e9, size=num_exp).tolist()
            for _ in range(num_repeats)
        ]

        for k in range(num_repeats):
            out_dir = os.path.join(out_parent_dir, f'repeat{k:02d}')
            sample_and_compute_params = dict(
                params=params.cross_val_params,               # <<< MODIFIED (LDA -> LDA/SVM OPTION)
                manifold_names=params.manifold_names,
                time_window=time_window,
                sample_size=sample_size
            )
            process_data_db_parallel(
                sample_and_compute_classifier,                # <<< MODIFIED (LDA -> LDA/SVM OPTION)
                exp_list,
                out_dir,
                in_dir,
                parallelism=params.parallelism,
                seeds=seeds[k],
                params=sample_and_compute_params,
            )
    return out_parent_dir


def read_dfs_from_dir(dist_dir, exp_list, num_repeats, metric, manifold_names):
    # Multiply sample numbers with exp_list
    keys = list(product(exp_list, range(num_repeats)))
    keys = [(exp[0], exp[1], k) for exp, k in keys]

    simdf_lists = []
    for exp_name, condition, k in keys:
        in_dir = os.path.join(dist_dir, f'repeat{k:02d}')
        simdf = read_df(in_dir, exp_name, verbose=False)
        simdf_mat = reshape_pair_df_to_matrix(simdf, metric, manifold_names)
        simdf_lists.append(simdf_mat)

    all_simdf = pd.concat(
        simdf_lists, keys=keys, names=['fish_id', 'condition', 'sample']
    )
    return all_simdf


def reshape_pair_df_to_matrix(pair_df, metric, manifold_names=None):
    # Rename the last column to metric
    pair_df = pair_df.rename(columns={pair_df.columns[-1]: metric, 'manifold1': 'odor', 'manifold2': 'manifold'})
    mat = pair_df.pivot_table(
        index='odor', columns='manifold', values=metric
    )
    if manifold_names is not None:
        mat = mat.reindex(index=manifold_names, columns=manifold_names)
    return mat


@dataclass_json
@dataclass
class RunLinearSeparabilityManifoldPairParams:  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    config_file: str
    compute_params: ComputeLinearSeparabilityManifoldPairParams  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    do_reorder_cs: bool
    odor_orders: list = None
    naive_name: str = 'naive'
    vs_same_ylim: list = None
    do_compare_cs: bool = False
    vsdict: dict = None
    plot_params: PlotDistanceParams = None


def run_linear_separability_manifold_pair(params: RunLinearSeparabilityManifoldPairParams):  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    dsconfig = load_dataset_config(params.config_file)
    exp_list = dsconfig.exp_list
    trace_dir = dsconfig.processed_trace_dir
    time_window = np.array(params.compute_params.time_window)  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    in_dir = trace_dir
    params.compute_params.in_dir = in_dir  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    params.compute_params.exp_list = exp_list  # <<< MODIFIED (LDA -> LDA/SVM OPTION)

    print('Plotting average trace...')
    fig_avg_trace, ax = plot_avg_trace_with_window(
        in_dir, exp_list[0][0], time_window
    )

    print('Computing distance matrices...')
    out_dir = compute_linear_separability_manifold_pair(params.compute_params)  # <<< MODIFIED (LDA -> LDA/SVM OPTION)

    all_dfs = read_dfs_from_dir(
        out_dir, exp_list, params.compute_params.num_repeats,  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
        params.compute_params.metric, dsconfig.odors_stimuli    # <<< MODIFIED (LDA -> LDA/SVM OPTION)
    )
    avg_simdf = average_over_repeats(all_dfs)

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
                params.compute_params.metric,  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
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
        params.compute_params.metric,  # <<< MODIFIED (LDA -> LDA/SVM OPTION)
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

    return output_figs, test_results
