import numpy as np
import pandas as pd


def group_and_flatten_responses(resp):
    resp_grouped = resp.T.groupby(level='condition', sort=False)
    respli = []
    for cond, group in resp_grouped:
        values = np.ravel(group.values)
        for resp in values:
            respli.append((cond, resp))

    resp_df = pd.DataFrame(respli, columns=['condition', 'response'])
    resp_df.set_index('condition', inplace=True)
    return resp_df


def normalize_responses(resp_df):
    # resp_df has one level index 'condtion'
    # select the 'naive' condition and normalize all responses to it
    baseline = resp_df.xs('naive').response.median()
    resp_df['normalized_response'] = (resp_df.response - baseline)/ baseline * 100
    return resp_df