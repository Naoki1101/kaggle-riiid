import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations

sys.path.append('../src')
import const
from feature_utils import save_features

lags = [1, 2, 3, 4, 5]
s = 1e-3


def get_features(df):
    features_df = pd.DataFrame()

    for lag in lags:
        features_df[f'timestamp_diff{lag}'] = np.zeros(len(df))

    user_gp = df[['user_id', 'timestamp']].groupby('user_id')

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        for lag in lags:
            features_df.loc[user_idx, f'timestamp_diff{lag}'] += user_df['timestamp'].diff(lag)

    for lag1, lag2 in list(combinations(lags, 2)):
        features_df[f'timestamp_diff{lag1}_div_timestamp_diff{lag2}'] = features_df[f'timestamp_diff{lag1}'] / (features_df[f'timestamp_diff{lag2}'] + s)

    features_df.columns = [f'dropped___{col}' for col in features_df.columns]

    return features_df


def main():
    train_df = pd.read_csv('../data/processed/train_dropped.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
