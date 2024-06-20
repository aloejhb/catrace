import numpy as np
import pandas as pd
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from os.path import join as pjoin

from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.manifold_analysis import manifold_analysis

from .utils import load_config
from .process_time_trace import SelectDfConfig, select_dataframe, sample_neuron, select_time_points, select_odors_df
from .exp_collection import read_df

def compute_mftma(dff, with_center_corr=False, kappa=0, n_t=200, n_reps=1):
    grouped = dff.groupby('odor', observed=True)
    manifolds = [group.T.to_numpy() for _, group in grouped]
    if with_center_corr:
        alpha_m, radius_m, dimension_m, res_coeff0, KK = manifold_analysis_corr(manifolds, kappa=kappa, n_t=n_t, n_reps=n_reps)
        ma_result = {'alpha_m': alpha_m,
                    'radius_m': radius_m,
                    'dimension_m': dimension_m,
                    'res_coeff0': res_coeff0,
                    'KK': KK}
    else:
        alpha_m, radius_m, dimension_m= manifold_analysis(manifolds, kappa, n_t)
        ma_result = {'alpha_m': alpha_m,
                    'radius_m': radius_m,
                    'dimension_m': dimension_m}
    return ma_result

def compute_ma_io(exp_name, in_dir, out_dir, ma_window=None,
                  sample_size=None, random_state=None,
                  odors=None,
                  with_center_corr=False, kappa=0):
    dff = read_df(in_dir, exp_name).dropna()
    if sample_size is not None:
        dff = sample_neuron(dff, sample_size=sample_size, random_state=random_state)
    dff = select_time_points(dff, ma_window)
    dff = select_odors_df(dff, odors)
    ma_result = compute_mftma(dff, with_center_corr=with_center_corr, kappa=kappa)
    np.savez(pjoin(out_dir, f'{exp_name}.npz'), **ma_result)

# def compute_mftma_io(input_file, config_file, output_file):
#     config = load_config(config_file, MftmaConfig)
#     dff = pd.read_pickle(input_file).dropna()
#     dff = select_dataframe(dff, config.select_df_config)
#     ma_result = compute_mftma(dff, kappa=config.kappa, n_t=config.n_t)
#     np.savez(output_file, **ma_result)
#     return ma_result

@dataclass_json
@dataclass
class MftmaConfig:
    select_df_config: SelectDfConfig
    kappa: float
    n_t: int # Number of gaussian vectors to sample per manifold