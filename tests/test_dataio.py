import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from catrace.dataio import save_tracedf_to_hdf5, load_tracedf_from_hdf5

@pytest.fixture
def sample_tracedf():
    # Create synthetic MultiIndex DataFrame
    odors = ["odorA", "odorB"]
    trials = [0, 1]
    times = np.arange(3)  # 3 frames
    neurons = [12, 13]

    index = pd.MultiIndex.from_product(
        [odors, trials, times],
        names=["odor", "trial", "time"]
    )
    columns = pd.MultiIndex.from_product(
        [neurons],
        names=["neuron"]
    )
    
    # Random data for testing
    data = np.random.rand(len(index), len(columns))
    return pd.DataFrame(data, index=index, columns=columns)

def test_hdf5_round_trip(sample_tracedf):
    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_data.h5")
        
        save_tracedf_to_hdf5(sample_tracedf, filepath)
        assert os.path.exists(filepath), "HDF5 file was not created"

        # Load back
        loaded_df = load_tracedf_from_hdf5(filepath)

        # Compare values
        assert np.all(loaded_df.values == sample_tracedf.values), "Loaded DataFrame does not match original DataFrame"
        assert np.all(loaded_df.index.to_frame().values == sample_tracedf.index.to_frame().values), "Loaded DataFrame index does not match original DataFrame index"
        assert np.all(loaded_df.columns.to_frame().values == sample_tracedf.columns.to_frame().values), "Loaded DataFrame columns do not match original DataFrame columns"
