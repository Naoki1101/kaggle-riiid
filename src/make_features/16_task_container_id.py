import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    user_gp = df[['user_id', 'content_id', 'task_container_id']].groupby(['user_id'])

    feat_array = np.zeros(len(df))
    feat_div_max_array = np.zeros(len(df))

    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_task_gp = user_df.groupby('task_container_id')

        for _, user_task_df in user_task_gp:
            user_task_idx = user_task_df.index.values
            if len(user_task_idx) == 1:
                feat_array[user_task_idx] = 1
                feat_div_max_array[user_task_idx] = 1.0
            else:
                order_array = np.argsort(np.argsort(user_task_df['content_id'])) + 1
                feat_array[user_task_idx] = order_array
                feat_div_max_array[user_task_idx] = order_array / order_array.max()

    features_df['order_content_id_by_task_container_id'] = feat_array
    features_df['order_content_id_div_max_by_task_container_id'] = feat_div_max_array

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
