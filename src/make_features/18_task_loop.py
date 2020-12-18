import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features
from utils import DataHandler

dh = DataHandler()


def add_user_feats(df, task_user_dict):
    task_nunique5_array = np.zeros(len(df), dtype=np.int8)
    task_nunique10_array = np.zeros(len(df), dtype=np.int8)
    default_list = np.repeat(np.nan, 10)

    for idx, row in enumerate(tqdm(df[['user_id', 'task_container_id']].values)):
        user_id = row[0]
        task_id = row[1]

        hist_array = task_user_dict.setdefault(user_id, default_list.copy())
        task_nunique5_array[idx] = np.unique(hist_array[-5:][~np.isnan(hist_array[-5:])]).shape[0]
        task_nunique10_array[idx] = np.unique(hist_array[~np.isnan(hist_array)]).shape[0]

        new_hist_list = list(task_user_dict[user_id])
        new_hist_list.pop(0)
        new_hist_list.append(task_id)
        task_user_dict[user_id] = np.array(new_hist_list)

    dh.save('../data/processed/task_user_dict.pkl', task_user_dict)

    user_feats_df = pd.DataFrame({
        'u_hist_task_nunique5': task_nunique5_array,
        'u_hist_task_nunique10': task_nunique10_array,
    })

    return user_feats_df


def get_features(df):
    features_df = pd.DataFrame()

    task_user_dict = {}

    features_df = add_user_feats(df, task_user_dict)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
