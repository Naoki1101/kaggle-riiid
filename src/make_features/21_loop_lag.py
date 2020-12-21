import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

SEQ_LEN = 100
s = 1e-3


def add_user_feats(df, user_timestamp_dict):
    user_difftime_median100 = np.zeros(len(df))
    user_difftime = np.zeros(len(df))
    default_array = np.repeat(np.nan, SEQ_LEN).astype(np.float64)

    for idx, row in enumerate(tqdm(df[['user_id', 'timestamp']].values)):
        user_id = row[0]
        timestamp = row[1]

        user_hist_timestamp_array = user_timestamp_dict.setdefault(user_id, default_array.copy())
        user_hist_difftime_array = np.diff(user_hist_timestamp_array)
        user_difftime_median100[idx] = np.median(user_hist_difftime_array[~np.isnan(user_hist_difftime_array)])
        user_difftime[idx] = timestamp - user_hist_timestamp_array[-1]

        new_timestamp_list = list(user_timestamp_dict[user_id])
        new_timestamp_list.pop(0)
        new_timestamp_list.append(timestamp)
        user_timestamp_dict[user_id] = np.array(new_timestamp_list)

    user_feats_df = pd.DataFrame({
        'user_difftime_median100': user_difftime_median100,
        'user_difftime': user_difftime,
    })

    user_feats_df['user_difftime_div_median100'] = user_feats_df['user_difftime'] / user_feats_df['user_difftime_median100']
    user_feats_df['user_difftime_diff_median100'] = user_feats_df['user_difftime'] - user_feats_df['user_difftime_median100']

    user_feats_df = user_feats_df.drop(['user_difftime'], axis=1)

    return user_feats_df


def get_features(df):
    features_df = pd.DataFrame()

    user_lag_dict = {}
    features_df = add_user_feats(df, user_lag_dict)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
