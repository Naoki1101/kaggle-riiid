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
        features_df[f'seq2dec_w{w}'] = np.zeros(len(df))

    user_gp = df[['user_id', 'answered_correctly']].groupby('user_id')

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index

        for window in windows:
            seq2dec_feats = user_df['answered_correctly'].shift(1)
            for lag in range(1, window):
                seq2dec_feats += user_df['answered_correctly'].shift(lag + 1) * (10 ** -lag)

            features_df.loc[user_idx, f'seq2dec_w{window}'] = seq2dec_feats

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
