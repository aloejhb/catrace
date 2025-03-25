import json
import os
import pandas as pd
import numpy as np
import pickle
import copy
import argparse

import gcmc
import gcmc.preprocess

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
        fig_avg_trace, ax = plot_avg_trace_with_window_dff(session, params.window)
        # Save the figure to current directory as png
        fig_avg_trace.savefig(outfile.replace('.pkl', '_avg_trace.png'))

    # select window
    session_in_window = select_time_points(session, params.window)
    session_in_window = select_odors_and_sort(session_in_window, params.odors)
    XtotT = get_manifolds(session_in_window)

    # Downsample neurons
    if params.N > 0:
        x_ind = np.random.choice(range(XtotT[0].shape[0]),params.N,replace=False)
        XtotT = [X[x_ind,:] for X in XtotT]
    else:
        print('N = 0, no downsampling')
        x_ind = np.arange(XtotT[0].shape[0])
        XtotT = [X[x_ind,:] for X in XtotT]


    # Preprocessing
    input_manifolds = gcmc.preprocess.standard_preprocessing(XtotT,global_center=params.global_center,bias=params.bias,
                                                             target_num_points_per_manifold=params.M)
    input_manifolds_shuffle = gcmc.preprocess.input_manifolds_shuffle(input_manifolds)


    # Gaussianize
    if params.gaussianize:
        input_manifolds_gaussianized = gcmc.preprocess.manifold_gaussianization(input_manifolds)
        input_manifolds_shuffle_gaussianized = gcmc.preprocess.manifold_gaussianization(input_manifolds_shuffle)
        input_manifolds_gaussianized = np.array(input_manifolds_gaussianized)
        input_manifolds_shuffle_gaussianized = np.array(input_manifolds_shuffle_gaussianized)

        input_manifolds = input_manifolds_gaussianized
        input_manifolds_shuffle = input_manifolds_shuffle_gaussianized

    # Check if the manifolds contains NaN
    for i in range(len(input_manifolds)):
        if np.isnan(input_manifolds[i]).any():
            print(f'input_manifolds[{i}] contains NaN')

 
    # Step 2: Conduct analysis
    df_list = []
    matrix_list = {'center_alignment_matrix': [], 'axes_alignment_matrix': [], 'center_axes_alignment_matrix': []}

    P = input_manifolds.shape[0]
    for i_P in range(P):
        input_manifolds_new = copy.deepcopy(input_manifolds)
        input_manifolds_new[[0,i_P]] = input_manifolds_new[[i_P,0]] # swap the 0-th manifold and the i_P-th manifold
        result = gcmc.manifold_analysis(input_manifolds_new,label_sample_type=gcmc.LabelSampleType.FIRST_ELEMENT_ONLY)
        filtered_result = {key: value[0] for key, value in result.items() if isinstance(value, np.ndarray) if value.shape == (1,) if value[0] != None}
        tuples = (fish_id,params.condition,params.window[0],params.window[1],False,i_P,params.seed, odors[0], odors[1])
        index = pd.MultiIndex.from_tuples([tuples], names=['fish_id','condition','window_0','window_1','shuffle','i_P','seed', 'odor_0', 'odor_1'])
        df_result = pd.DataFrame(filtered_result, index=index)
        df_list.append(df_result)

        input_manifolds_shuffle_new = copy.deepcopy(input_manifolds_shuffle)
        input_manifolds_shuffle_new[[0,i_P]] = input_manifolds_shuffle_new[[i_P,0]] # swap the 0-th manifold and the i_P-th manifold
        result = gcmc.manifold_analysis(input_manifolds_shuffle_new,label_sample_type=gcmc.LabelSampleType.FIRST_ELEMENT_ONLY)
        filtered_result = {key: value[0] for key, value in result.items() if isinstance(value, np.ndarray) if value.shape == (1,) if value[0] != None}
        tuples = (fish_id,params.condition,params.window[0],params.window[1],True,i_P,params.seed, odors[0], odors[1])
        index = pd.MultiIndex.from_tuples([tuples], names=['fish_id','condition','window_0','window_1','shuffle','i_P','seed', 'odor_0', 'odor_1'])
        df_result = pd.DataFrame(filtered_result, index=index)
        df_list.append(df_result)

    df_result = pd.concat(df_list)
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
   