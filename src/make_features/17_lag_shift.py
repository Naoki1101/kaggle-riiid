import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

s = 1e-3


def get_features(df):
    features_df = pd.DataFrame()

    user_gp = df[['user_id', 'timestamp']].groupby('user_id')

    lag_feats_array = np.zeros(len(df))

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        lag_feats_array[user_idx] = user_df['timestamp'].diff(1).shift(1)

    features_df['timestamp_lag_shift1'] = lag_feats_array

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
