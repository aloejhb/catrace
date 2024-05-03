import pytest
import numpy as np
import pandas as pd
import unittest
from .conftest import create_test_dataframe
from catrace.utils import save_config
from catrace.mft_manifold_analysis import compute_mftma, compute_mftma_io, MftmaConfig
from catrace.process_time_trace import SelectDfConfig

def test_compute_mftma():
    dff = create_test_dataframe()
    results = compute_mftma(dff)
    assert isinstance(results, dict)
    expected_keys = ['alpha_m', 'radius_m', 'dimension_m', 'res_coeff0', 'KK']
    for key in expected_keys:
        assert key in results

def test_compute_mftma_io(tmp_path):
    config_file = tmp_path / "config.json"
    input_file = tmp_path / "input.pkl"
    output_file = tmp_path / "output.npz"

    config = MftmaConfig(select_df_config=SelectDfConfig(odors=['banana', 'apple', 'date'], time_window=[3, 10], sort=True), kappa=0.01, n_t=200)
    save_config(config, config_file)

    dff = create_test_dataframe()
    dff.to_pickle(input_file)

    result = compute_mftma_io(str(input_file), str(config_file), str(output_file))

    assert isinstance(result, dict)
    assert 'alpha_m' in result
    assert output_file.exists()


