def match_template(df, frame_window=[35, 47]):
    df_trunc = ptt.truncate_binned_df(df, frame_window)
    df_response = df_trunc.groupby(level=['odor', 'trial']).mean()
    df_response = df_response[~df_response.index.get_level_values('odor').isin(['acsf', 'spont'])]

    template = df_response.xs(slice(0, 1), level='trial', axis=0).groupby(level=['odor']).mean()
    test_df = df_response.xs(2, level='trial')
    odor_labels = test_df.apply(lambda row: find_closest(row, template), axis=1)
    return odor_labels

# # Calculate the number of incorrect predictions
# incorrect_predictions = sum(df_trial3.index.get_level_values('odor') != df_trial3['predicted_odor'])

# # Calculate the total number of predictions
# total_predictions = len(df_trial3)

# # Calculate the error rate
# error_rate = incorrect_predictions / total_predictions

# print('Error Rate:', error_rate)
