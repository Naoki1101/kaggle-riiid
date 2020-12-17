import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    user_gp = df[['user_id', 'content_id', 'task_container_id']].groupby('user_id')

    content_id_diff_array = np.zeros(len(df))
    task_container_id_diff_array = np.zeros(len(df))

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index.values

        content_id_diff_array[user_idx] = (user_df['content_id'] - user_df['content_id'].shift(1)).abs()
        task_container_id_diff_array[user_idx] = (user_df['task_container_id'] - user_df['task_container_id'].shift(1)).abs()

    features_df['content_id_diff_abs'] = content_id_diff_array
    features_df['task_container_id_diff_abs'] = task_container_id_diff_array

    features_df.columns = [f'dropped___{col}' for col in features_df.columns]

    return features_df


def main():
    train_df = pd.read_csv('../data/processed/train_dropped.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
