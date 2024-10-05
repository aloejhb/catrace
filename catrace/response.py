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
    return resp_df


# resp_df['normalized_response'] = resp_df.response / resp_df[resp_df.cond == 'naive'].response.median() * 100