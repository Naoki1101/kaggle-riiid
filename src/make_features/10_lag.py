import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

lags = [1, 2, 3, 4, 5]


def get_features(df):
    features_df = pd.DataFrame()

    for lag in lags:
        features_df[f'timestamp_diff{lag}'] = np.zeros(len(df))

    user_gp = df.groupby('user_id')

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        for lag in lags:
            features_df.loc[user_idx, f'timestamp_diff{lag}'] += user_df['timestamp'].diff(lag)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
