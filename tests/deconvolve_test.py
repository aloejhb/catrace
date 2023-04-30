"""Tests for deconvolve."""

from absl.testing import absltest
import os
import numpy as np
import pandas as pd
from catrace import deconvolve
from cascade2p import cascade


def generate_trace_df():
    odor = ['odor1', 'odor2', 'odor3']
    trial = [0, 1]
    time = range(12)

    dataframes = []
    for o, t, tp in pd.MultiIndex.from_product([odor, trial, time], names=['odor', 'trial', 'time']):
        matrix = np.ones((1, 5)) + np.random.rand(1, 5) * 0.2
        df = pd.DataFrame(matrix, columns=[0, 1, 2, 3, 4])
        df.index = pd.MultiIndex.from_tuples([(o, t, tp)] * 1, names=['odor', 'trial', 'time'])
        dataframes.append(df)

    trace_df = pd.concat(dataframes)
    return trace_df


class DeconvolveTest(absltest.TestCase):

    def test_deconvolve_experiment(self):
        trace_df = generate_trace_df()
        sampling_rate = 7.5
        baseline_window = (0, 35)
        model_name = 'Global_EXC_7.5Hz_smoothing200ms_causalkernel'
        cascade_dir = os.path.dirname(cascade.__file__)
        model_folder = os.path.join(cascade_dir, '..', 'Pretrained_models')
        spike_prob_df = deconvolve.deconvolve_experiment(trace_df,
                                                         model_name,
                                                         model_folder,
                                                         sampling_rate,
                                                         baseline_window)
        # TODO create more realistic trace and verify results


if __name__ == '__main__':
    absltest.main()
