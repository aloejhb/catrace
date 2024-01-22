import json
import pandas as pd


def copy_index(df1, df2):
    """
    Copy index of df1 to df2 while keep the original df2 index as one level
    """
    df1_index = df1.index.to_frame(index=False)
    df2_index = df2.index.to_frame(index=False)
    dfidx = df2_index.join(df1_index)
    idx = pd.MultiIndex.from_frame(dfidx)
    df2 = df2.set_index(idx)
    return df2


def copy_frame_structure(arr, df):
    """
    Copy the row and column indices and create a new dataframe with values of an array
    """
    assert df.shape == arr.shape, "Dataframe and array shapes should be the same"
    new_df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    return new_df


def load_config(file_path, config_class):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        config = config_class.from_json(json_data)
        return config
