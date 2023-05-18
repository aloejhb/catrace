from scipy.spatial import distance
import catrace.process_time_trace as ptt

def match_template(df, frame_window=[35, 47], metric='euclidean'):
    df_trunc = ptt.truncate_df_window(df, frame_window)
    df_response = df_trunc.groupby(level=['odor', 'trial']).mean()
    df_response = df_response[~df_response.index.get_level_values('odor').isin(['acsf', 'spont'])]

    template = df_response.xs(slice(0, 1), level='trial', axis=0).groupby(level=['odor']).mean()
    test_df = df_response.xs(2, level='trial')
    odor_labels = test_df.apply(lambda row: find_closest(row, template), axis=1)
    return odor_labels


def find_closest(row, templates, metric='euclidean'):
    if metric == 'euclidean':
        distance_func = distance.euclidean
    elif metric == 'cosine':
        distance_func = distance.cosine
    else:
        raise ValueError('Metric should be either euclidean for cosine')
    distances = templates.apply(lambda x: distance_func(x, row), axis=1)
    return distances.idxmin()


# # Calculate the number of incorrect predictions
# incorrect_predictions = sum(df_trial3.index.get_level_values('odor') != df_trial3['predicted_odor'])

# # Calculate the total number of predictions
# total_predictions = len(df_trial3)

# # Calculate the error rate
# error_rate = incorrect_predictions / total_predictions

# print('Error Rate:', error_rate)
