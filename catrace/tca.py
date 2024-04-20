import pandas as pd


def reshape_to_3d(dff: pd.DataFrame):
    df_unstacked = dff.unstack(level='time', sort=False)
    array_3d = df_unstacked.to_numpy()
    num_trials = len(df_unstacked)
    num_neurons = len(dff.columns)
    num_timepoints = len(dff.index.unique('time'))
    array_3d_reshaped = array_3d.reshape(num_trials, num_neurons, num_timepoints)
    return array_3d_reshaped