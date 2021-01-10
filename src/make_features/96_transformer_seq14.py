import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.append('../src')
import const
from utils import DataHandler

dh = DataHandler()
MAX_SEQ = 201
usecols = ['row_id', 'user_id', 'content_id', 'timestamp',
           'prior_question_elapsed_time', 'part', 'answered_correctly',
           'user_answer', 'task_container_id', 'is_val']


def add_user_feats(df, user_content_dict, user_timestamp_dict, user_prior_elapsed_dict,
                   user_part_dict, user_target_dict, user_ans_dict, user_task_dict):
    for idx, row in enumerate(tqdm(df[usecols].values)):
        row_id = row[0]
        user_id = row[1]
        content_id = row[2]
        timestamp = row[3]
        prior_elapsed = row[4]
        part = row[5]
        target = row[6]
        user_ans = row[7]
        task_id = row[8]
        val_flg = row[9]

        update_dict(user_id, user_content_dict, content_id)
        update_dict(user_id, user_timestamp_dict, timestamp)
        update_dict(user_id, user_prior_elapsed_dict, prior_elapsed)
        update_dict(user_id, user_part_dict, part)
        update_dict(user_id, user_target_dict, target)
        update_dict(user_id, user_ans_dict, user_ans)
        update_dict(user_id, user_task_dict, task_id)

        if val_flg:
            seq_list = [
                np.array(user_content_dict[user_id][1:]),
                np.diff(user_timestamp_dict[user_id]),
                np.array(user_prior_elapsed_dict[user_id][1:]),
                np.array(user_part_dict[user_id][1:]),
                np.array(user_target_dict[user_id][:-1]),
                np.array(user_ans_dict[user_id][:-1]),
                np.array(user_task_dict[user_id][1:])
            ]

            save_seq(row_id, seq_list)


def get_features(df):
    user_content_dict = {}
    user_timestamp_dict = {}
    user_prior_elapsed_dict = {}
    user_part_dict = {}
    user_target_dict = {}
    user_ans_dict = {}
    user_task_dict = {}
    add_user_feats(df, user_content_dict, user_timestamp_dict, user_prior_elapsed_dict,
                   user_part_dict, user_target_dict, user_ans_dict, user_task_dict)


def update_dict(user_id, user_dict, v):
    default_list = [0] * MAX_SEQ

    if user_id not in user_dict:
        user_dict[user_id] = default_list.copy()

    new_list = user_dict[user_id]
    new_list.pop(0)
    new_list.append(v)
    user_dict[user_id] = new_list


def save_seq(row_id, seq_list):
    seq_dir = Path('../data/seq14')
    if not seq_dir.exists():
        seq_dir.mkdir(exist_ok=True)

    dh.save(seq_dir / f'row_{int(row_id)}.pkl', seq_list)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    questions_df = pd.read_csv('../data/input/questions.csv')
    q2p = dict(questions_df[['question_id', 'part']].values)
    train_df['part'] = train_df['content_id'].map(q2p)
    train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(float)

    val_idx = np.load('../data/processed/cv1_valid_v2.npy')
    train_df['is_val'] = False
    train_df.loc[val_idx, 'is_val'] = True

    train_df = train_df[train_df['content_type_id'] == 0].reset_index(drop=True)

    get_features(train_df)


if __name__ == '__main__':
    main()