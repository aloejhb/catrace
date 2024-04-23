import pytest
import numpy as np
import pandas as pd
from catrace.process_time_trace import select_dataframe, SelectDfConfig
from .conftest import create_test_dataframe

def test_select_dataframe():
    dff = create_test_dataframe()
    config = SelectDfConfig(odors=['cherry', 'apple'], time_window=[2, 4], sort=False)
    result_df = select_dataframe(dff, config)
    assert all(item in result_df.index.unique('odor') for item in ['apple', 'cherry'])

def test_select_dataframe_with_sort():
    dff = create_test_dataframe()
    config = SelectDfConfig(odors=['cherry', 'apple'], time_window=[2, 4], sort=False)
    result_df = select_dataframe(dff, config)
    assert all(item in result_df.index.unique('odor') for item in ['cherry', 'apple'])