import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from utils import DataHandler
from feature_utils import save_features

lags = [1, 2, 3, 4, 5]
s = 1e-3
dh = DataHandler()


def get_features(df):
    features_df = pd.DataFrame()

    features_df['timestamp_lag_mean'] = np.zeros(len(df))
    features_df['timestamp_lag_diff_mean_each_user'] = np.zeros(len(df))
    features_df['timestamp_lag_div_mean_each_user'] = np.zeros(len(df))

    user_gp = df[['user_id', 'timestamp']].groupby('user_id')

    user_lag_mean_dict = {}
    user_lag_median_dict = {}

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        lag_feats = user_df['timestamp'].diff()

        features_df.loc[user_idx, 'timestamp_lag_mean'] = lag_feats.mean()
        features_df.loc[user_idx, 'timestamp_lag_diff_mean_each_user'] = lag_feats - lag_feats.mean()
        features_df.loc[user_idx, 'timestamp_lag_div_mean_each_user'] = lag_feats / (lag_feats.mean() + s)
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling3_mean_each_user'] = lag_feats - (lag_feats.rolling(4).sum() - lag_feats) / 3
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling5_mean_each_user'] = lag_feats - (lag_feats.rolling(6).sum() - lag_feats) / 5
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling7_mean_each_user'] = lag_feats - (lag_feats.rolling(8).sum() - lag_feats) / 7

        features_df.loc[user_idx, 'timestamp_lag_median'] = lag_feats.median()
        features_df.loc[user_idx, 'timestamp_lag_diff_median_each_user'] = lag_feats - lag_feats.median()
        features_df.loc[user_idx, 'timestamp_lag_div_median_each_user'] = lag_feats / (lag_feats.median() + s)
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling3_median_each_user'] = lag_feats - lag_feats.rolling(4).median()
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling5_median_each_user'] = lag_feats - lag_feats.rolling(6).median()
        features_df.loc[user_idx, 'timestamp_lag_diff_rolling7_median_each_user'] = lag_feats - lag_feats.rolling(8).median()

        user_lag_mean_dict[user_id] = lag_feats.mean()
        user_lag_median_dict[user_id] = lag_feats.median()

    dh.save(f'../data/processed/dropped___user_lag_mean_encoder.pkl', user_lag_mean_dict)
    dh.save(f'../data/processed/dropped___user_lag_median_encoder.pkl', user_lag_median_dict)

    features_df.columns = [f'dropped___{col}' for col in features_df.columns]

    return features_df


def main():
    train_df = pd.read_csv('../data/processed/train_dropped.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
