import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

lags = [1, 2, 3, 4, 5]
s = 1e-3


def get_features(df):
    features_df = pd.DataFrame()

    features_df['timestamp_lag_mean'] = np.zeros(len(df))
    features_df['timestamp_lag_diff_mean_each_user'] = np.zeros(len(df))
    features_df['timestamp_lag_div_mean_each_user'] = np.zeros(len(df))

    user_gp = df[['user_id', 'timestamp']].groupby('user_id')

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        lag_feats = user_df['timestamp'].diff()

        features_df.loc[user_idx, 'timestamp_lag_mean'] = np.mean(lag_feats)
        features_df.loc[user_idx, 'timestamp_lag_diff_mean_each_user'] = lag_feats - np.mean(lag_feats)
        features_df.loc[user_idx, 'timestamp_lag_div_mean_each_user'] = lag_feats / (np.mean(lag_feats) + s)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
