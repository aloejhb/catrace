import pytest
import numpy as np
import pandas as pd
from catrace.process_time_trace import select_dataframe, SelectDfConfig

def create_test_dataframe():
    odors = ['apple', 'banana', 'cherry', 'date']
    trials = [0, 1, 2]
    times = list(range(6))
    index = pd.MultiIndex.from_product([odors, trials, times], names=['odor', 'trial', 'time'])
    values = np.random.randint(1, 10, size=(len(odors) * len(trials) * len(times)))
    dff = pd.DataFrame({'value': values}, index=index)
    return dff

def test_select_dataframe():
    dff = create_test_dataframe()
    print(dff)

    config = SelectDfConfig(odors=['apple', 'cherry'], time_window=['t1', 't3'], sort=False)

    result_df = select_dataframe(df, config)
    assert all(item in result_df['odor'].tolist() for item in ['apple', 'cherry'])

def test_select_dataframe_with_sort():
    pass