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
