import sys
import random
import numpy as np
import pandas as pd

sys.path.append('../src')
import const
from utils import seed_everything

val_size = 2_500_000


# https://www.kaggle.com/its7171/cv-strategy
def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    # max_timestamp_u = train_df[['user_id', 'timestamp']].groupby(['user_id']).agg(['max']).reset_index()
    # max_timestamp_u.columns = ['user_id', 'max_time_stamp']
    # MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

    # def rand_time(max_time_stamp):
    #     interval = MAX_TIME_STAMP - max_time_stamp
    #     rand_time_stamp = random.randint(0, interval)
    #     return rand_time_stamp

    # max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
    # train_df = train_df.merge(max_timestamp_u, on='user_id', how='left')
    # train_df['viretual_time_stamp'] = train_df.timestamp + train_df['rand_time_stamp']

    # train_df = train_df.sort_values(['viretual_time_stamp', 'row_id'])

    # sorted_idx = train_df.index
    # valid_cv1_idx = sorted_idx[-val_size:].values

    cv1_valid = pd.read_pickle('../folds/cv1_valid.pickle')
    valid_row_id = cv1_valid['row_id']
    valid_cv1_idx = train_df[train_df['row_id'].isin(valid_row_id)].index.values

    np.save('../pickle/cv1_valid.npy', valid_cv1_idx)


if __name__ == '__main__':
    main()
