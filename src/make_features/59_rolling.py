import sys
import numpy as np
import pandas as pd

sys.path.append('../src')
import const
from feature_utils import save_features

windows = [3, 5, 7]


def get_features(df):
    features_df = pd.DataFrame()

    for w in windows:
        features_df[f'answered_correctly_rolling{w}'] = np.zeros(len(df))

    for user_id, user_df in df.groupby('user_id'):
        user_idx = user_df.index

        for w in windows:
            rolling_feats = (user_df['answered_correctly'].rolling(window=w + 1).sum()
                             - user_df['answered_correctly']) / w
            features_df.loc[user_idx, f'answered_correctly_rolling{w}'] += rolling_feats

    features_df.columns = [f'dropped___{col}' for col in features_df.columns]

    return features_df


def main():
    train_df = pd.read_csv('../data/processed/train_dropped.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
