import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

windows = [3, 5, 7]


def get_features(df):
    features_df = pd.DataFrame()

    for w in windows:
        features_df[f'answered_correctly_rolling{w}_div_elapsed_from_prior'] = np.zeros(len(df))

    user_gp = df.groupby('user_id')

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        for w in windows:
            rolling_feats = (user_df['answered_correctly'].rolling(window=w + 1).sum()
                             - user_df['answered_correctly']) / w

            time_diff = user_df['timestamp'].diff()

            features_df.loc[user_idx, f'answered_correctly_rolling{w}_div_elapsed_from_prior'] += rolling_feats / np.log1p(time_diff)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
