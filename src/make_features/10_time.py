import sys
import numpy as np
import pandas as pd

sys.path.append('../src')
import const
from feature_utils import save_features

lag_list = [1]


def get_features(df):
    features_df = pd.DataFrame()

    features_df[f'timestamp_diff1'] = np.zeros(len(df))

    for user_id, user_df in df.groupby('user_id'):
        user_idx = user_df.index

        features_df.loc[user_idx, f'timestamp_diff1'] += user_df['timestamp'].diff()

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
