import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from mftma.manifold_analysis_correlation import manifold_analysis_corr

from .utils import load_config
from .process_time_trace import SelectDfConfig, select_dataframe
from .exp_collection import read_df

def compute_mftma(dff, kappa=0):
    grouped = dff.groupby('odor', observed=True)
    manifolds = [group.T.to_numpy() for _, group in grouped]
    alpha_m, radius_m, dimension_m, res_coeff0, KK = manifold_analysis_corr(manifolds, kappa, 200, n_reps=1)
    ma_result = {'alpha_m': alpha_m,
                'radius_m': radius_m,
                'dimension_m': dimension_m,
                'res_coeff0': res_coeff0,
                'KK': KK}
    return ma_result

def compute_mftma_experiment(input_file, config_file, output_file):
    config = load_config(config_file, MftmaConfig)
    dff = read_df(input_file).dropna()
    dff = select_dataframe(dff, config.select_df_config)
    ma_result = compute_mftma(dff, kappa=kappa)
    np.savez(output_file, **ma_result)
    return ma_result

@dataclass_json
@dataclass
class MftmaConfig:
    self.select_df_config: SelectDfConfig
    self.kappa: float
    self.n_t: float # Number of gaussian vectors to sample per manifold