import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale


def standardize_all(df):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
    scaler.fit(df)
    df_centered = scaler.transform(df)
    df_scaled = df_centered / df_centered.std(ddof=1)
    df_out = pd.DataFrame(data=df_scaled, columns=df.columns, index=df.index)
    return df_out


def standard_scale(df):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_out = pd.DataFrame(data=df_scaled, columns=df.columns, index=df.index)
    return df_out


def min_max_scale(df):
    df_scaled = minmax_scale(df)
    df_out = pd.DataFrame(data=df_scaled, columns=df.columns, index=df.index)
    return df_out


def quantile_all(df, q=0.999):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
    scaler.fit(df)
    df_centered = scaler.transform(df)
    df_scaled = df_centered / np.quantile(np.absolute(df_centered),q=q)
    df_out = pd.DataFrame(data=df_scaled, columns=df.columns, index=df.index)
    return df_out
