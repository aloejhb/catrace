import json
import os
import pandas as pd
import numpy as np
import pickle
import copy
import argparse

import gcmc
from gcmc.contrib import gcmc_analysis_dataframe

from typing import Union

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from catrace.utils import load_json_to_dataclass
from catrace.process_time_trace import select_time_points, select_odors_and_sort

from typing import Optional, List, Tuple


@dataclass_json
@dataclass
class AnalysisParams:
    outfile: str
    indir: str
    fish_id: str
    condition: str
    window: Tuple[int, int]
    N: int
    M: int
    global_center: bool
    bias: bool
    seed: int
    repeat: int
    odors: Optional[List[str]] = None
    overwrite_computation: bool = False
    debug: bool = False
    gaussianize: bool = False


def get_manifolds(dff):
    manifolds = []
    for odor, group in dff.groupby(level='odor', sort=False, observed=True):
        manifold = group.T.to_numpy()
        manifolds.append(manifold)
    return manifolds


def analysis_one_vs_one(params: AnalysisParams):
    fish_id = params.fish_id
    outfile = params.outfile
    # Make sure the output directory exists
    outdir = os.path.dirname(params.outfile)
    os.makedirs(outdir, exist_ok=True)

    # Select odor
    if params.odors is None:
        raise ValueError('params.odors must be specified')
    if len(params.odors) != 2:
        raise ValueError('params.odors must contain two odors')
    # if two odors are the same, raise an error
    if params.odors[0] == params.odors[1]:
        raise ValueError('params.odors must contain two different odors')
    odors = params.odors
    odor1 = odors[0]
    odor2 = odors[1]

    # outfile should not be empty
    if not outfile:
        raise ValueError('params.outfile must be specified')

    if not params.overwrite_computation and os.path.exists(outfile):
        print(f'{outfile} already exists, skipping')
        return

    # load manifold
    session = pd.read_pickle(os.path.join(params.indir, f'{fish_id}.pkl'))
    # Make sure session contains two odors

    if params.debug:
        from catrace.run.run_utils import plot_avg_trace_with_window_dff

        fig_avg_trace, ax = plot_avg_trace_with_window_dff(
            session, params.window
        )
        # Save the figure to current directory as png
        fig_avg_trace.savefig(outfile.replace('.pkl', '_avg_trace.png'))

    # select window
    session_in_window = select_time_points(session, params.window)
    session_in_window = select_odors_and_sort(session_in_window, params.odors)
    XtotT = get_manifolds(session_in_window)

    # Downsample neurons
    if params.N > 0:
        x_ind = np.random.choice(
            range(XtotT[0].shape[0]), params.N, replace=False
        )
        XtotT = [X[x_ind, :] for X in XtotT]
    else:
        print('N = 0, no downsampling of neurons')
        x_ind = np.arange(XtotT[0].shape[0])
        XtotT = [X[x_ind, :] for X in XtotT]

    # Downsample time points
    XtotT = [
        X[:, np.random.choice(range(X.shape[1]), params.M, replace=False)]
        for X in XtotT
    ]

    # Check if the manifolds contains NaN
    for i in range(len(XtotT)):
        if np.isnan(XtotT[i]).any():
            print(f'input_manifolds[{i}] contains NaN')

    # This is a place holder until the centering and bias is clear to BoHu
    assert params.global_center == True
    assert params.bias == True

    # Step 2: Conduct analysis
    P = XtotT[0].shape[0]
    df_result = gcmc_analysis_dataframe(
        XtotT,
        (
            fish_id,
            params.condition,
            odor1,
            odor2,
            params.window[0],
            params.window[1],
        ),
        ['fish_id', 'condition', 'odor1', 'odor2', 'window_0', 'window_1'],
        seed=params.seed,
        n_hyperplanes=1000,
        shuffle=True,
        analysis_type='FIRST_VERSUS_REST',
        scale=1,
        geometry=True,
        gaussianize=params.gaussianize,
    )

    # Step 3: Save the results
    print(f'writing result to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(df_result, f)
    if params.debug:
        print(df_result)


def main(args: Union[argparse.Namespace, dict]):
    if isinstance(args, dict):
        params = AnalysisParams.from_dict(args)
    else:
        params = load_json_to_dataclass(args.params, AnalysisParams)

    print('running with params:', params)
    analysis_one_vs_one(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params')
    args = parser.parse_args()
    main(args)
