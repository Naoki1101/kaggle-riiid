import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append('../src')
import const
from utils import DataHandler

dh = DataHandler()
MAX_SEQ = 101
usecols = ['row_id', 'user_id', 'content_id', 'task_container_id', 'timestamp', 'part', 'answered_correctly', 'is_val']


def add_user_feats(df, user_content_dict, user_task_dict, user_timestamp_dict,
                   user_part_dict, user_target_dict):
    for idx, row in enumerate(tqdm(df[usecols].values)):
        row_id = row[0]
        user_id = row[1]
        content_id = row[2]
        task_container_id = row[3]
        timestamp = row[4]
        part = row[5]
        target = row[6]
        val_flg = row[7]

        update_dict(user_id, user_content_dict, content_id)
        update_dict(user_id, user_task_dict, task_container_id)
        update_dict(user_id, user_timestamp_dict, timestamp)
        update_dict(user_id, user_part_dict, part)
        update_dict(user_id, user_target_dict, target)

        if val_flg:
            seq_list = [
                user_content_dict[user_id][1:],
                user_task_dict[user_id][1:],
                np.diff(user_timestamp_dict[user_id]),
                user_part_dict[user_id][1:],
                user_target_dict[user_id][:-1],
            ]

            save_seq(row_id, seq_list)


def get_features(df):
    user_content_dict = {}
    user_task_dict = {}
    user_timestamp_dict = {}
    user_part_dict = {}
    user_target_dict = {}
    add_user_feats(df, user_content_dict, user_task_dict, user_timestamp_dict,
                   user_part_dict, user_target_dict)


def update_dict(user_id, user_dict, v):
    default_list = np.zeros(MAX_SEQ)

    if user_id not in user_dict:
        user_dict[user_id] = default_list.copy()

    new_list = list(user_dict[user_id])
    new_list.pop(0)
    new_list.append(v)
    user_dict[user_id] = np.array(new_list)


def save_seq(row_id, seq_list):
    seq_dir = Path('../data/seq2')
    if not seq_dir.exists():
        seq_dir.mkdir(exist_ok=True)

    dh.save(seq_dir / f'row_{int(row_id)}.pkl', seq_list)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    questions_df = pd.read_csv('../data/input/questions.csv')
    q2p = dict(questions_df[['question_id', 'part']].values)
    train_df['part'] = train_df['content_id'].map(q2p)

    val_idx = np.load('../data/processed/cv1_valid.npy')
    train_df['is_val'] = False
    train_df.loc[val_idx, 'is_val'] = True

    train_df = train_df[train_df['content_type_id'] == 0].reset_index(drop=True)

    get_features(train_df)


if __name__ == '__main__':
    main()
