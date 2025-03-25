import json
import os
import pandas as pd
import numpy as np
import pickle


import sys
sys.path.append("/mnt/home/cchou/Documents/Research/Correlation_geometry/gcmc")
import gcmc
import gcmc.data
import gcmc.preprocess
from gcmc.contrib import gcmc_analysis_dataframe
from gcmc import gcmc_analysis
from gcmc.types import ResultReportLevel



# Select the time window
def select_timepoints(df, window):
    """Select trace dataframe in a given time window"""
    df_filtered = df[(df.index.get_level_values('time') >= window[0])
                & (df.index.get_level_values('time') <= window[1])]
    return df_filtered

def get_manifolds(dff):
    manifolds = []
    for odor, group in dff.groupby(level='odor', sort=False):
        manifold = group.T.to_numpy()
        manifolds.append(manifold)
    return manifolds



def analysis_5(params):

    # Step 1: Load parameters and data
    outdir = params['outdir']
    subdir_name = params['subdir_name']
    dev_stage = params['dev_stage']
    fish_id = params['fish_id']
    condition = params['condition']
    window = params['window']
    odor1 = params['odor1']
    odor2 = params['odor2']
    N = params['N']
    M = params['M']
    global_center = params['global_center']
    bias = params['bias']
    num_rep = params['num_rep']
    seed = params['seed']

    home_dir = '/mnt/home/cchou/ceph/zebrafish_olfactory_manifolds'
    dataset_dir = os.path.join(home_dir,'data', 'full_datasets', dev_stage)
    trace_dir = os.path.join(dataset_dir, subdir_name)

    # load manifold
    session = pd.read_pickle(os.path.join(trace_dir, f'{fish_id}.pkl'))

    # select window
    session_in_window = select_timepoints(session, window)
    XtotT = [get_manifolds(session_in_window.query(f'odor=="{odor1}"'))[0],
        get_manifolds(session_in_window.query(f'odor=="{odor2}"'))[0]]

    df_list = []
    for _ in range(num_rep):
        # downsample neurons
        x_ind = np.random.choice(range(XtotT[0].shape[0]),N,replace=False)
        XtotT = [X[x_ind,:] for X in XtotT]

        # downsample points
        # XtotT = [X[:,np.random.choice(range(X.shape[1]),M,replace=False)] for X in XtotT]
        XtotT = [X[:,np.random.choice(range(X.shape[1]),M,replace=False)] * 7.5 for X in XtotT]

        df_result = gcmc_analysis_dataframe(
            XtotT,
            (fish_id,condition,odor1,odor2,window[0],window[1]),
            ['fish_id','condition','odor1','odor2','window_0','window_1'],
            seed=seed,
            n_hyperplanes=1000,
            shuffle=False,
            analysis_type='FIRST_VERSUS_REST',
            scale=1,
            geometry=True,
        )
        df_list.append(df_result)
        seed = seed + 1

    df_result = pd.concat(df_list)

    # Step 3: Save the results
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{os.environ['DISBATCH_REPEAT_INDEX']}" + '.pkl')
    print(f'writing result to {outfile}', flush=True)
    with open(outfile, 'wb') as f:
        pickle.dump(df_result, f)


def main(params):
    print('running with params:', params)
    if params['analysis_type'] == 5:
        analysis_5(params)
    else:
        print(f"Analysis type {params['analysis_type']} does not exist :(")
