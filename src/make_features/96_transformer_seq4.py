import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from utils import DataHandler

dh = DataHandler()
MAX_SEQ = 101
usecols = ['row_id', 'user_id', 'content_id', 'part', 'type_of_id', 'answered_correctly', 'is_val']


def get_features(df):
    user_content_dict = {}
    user_part_dict = {}
    user_type_dict = {}
    user_target_dict = {}
    add_user_feats(df, user_content_dict, user_type_dict, user_part_dict, user_target_dict)


def add_user_feats(df, user_content_dict, user_type_dict, user_part_dict, user_target_dict):
    for idx, row in enumerate(tqdm(df[usecols].values)):
        row_id = row[0]
        user_id = row[1]
        content_id = row[2]
        part = row[3]
        type_of = row[4]
        target = row[5]
        val_flg = row[6]

        update_dict(user_id, user_content_dict, content_id)
        update_dict(user_id, user_part_dict, part)
        update_dict(user_id, user_type_dict, type_of)
        update_dict(user_id, user_target_dict, target)

        if val_flg:
            seq_list = [
                user_content_dict[user_id][1:],
                user_part_dict[user_id][1:],
                user_type_dict[user_id][1:],
                user_target_dict[user_id][:-1],
            ]

            save_seq(row_id, seq_list)


def update_dict(user_id, user_dict, v):
    default_list = np.zeros(MAX_SEQ)

    if user_id not in user_dict:
        user_dict[user_id] = default_list.copy()

    new_list = list(user_dict[user_id])
    new_list.pop(0)
    new_list.append(v)
    user_dict[user_id] = np.array(new_list)


def save_seq(row_id, seq_list):
    dh.save(f'../data/seq4/row_{int(row_id)}.pkl', seq_list)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    questions_df = pd.read_csv('../data/input/questions.csv')
    lectures_df = pd.read_csv('../data/input/lectures.csv')

    lectures_df['type_of_id'] = lectures_df['type_of'].map({'concept': 0, 'solving question': 1, 'intention': 2, 'starter': 3})

    q2p = dict(questions_df[['question_id', 'part']].values)
    l2p = dict(lectures_df[['lecture_id', 'part']].values)
    q2p.update(l2p)
    train_df['part'] = train_df['content_id'].map(q2p) - 1

    l2t = dict(lectures_df[['lecture_id', 'type_of_id']].values)
    train_df['type_of_id'] = train_df['content_id'].map(l2t).fillna(4)

    lec_idx = train_df[train_df['content_type_id'] == 1].index
    train_df.loc[lec_idx, 'answered_correctly'] = 2

    val_idx = np.load('../data/processed/cv1_valid.npy')
    train_df['is_val'] = False
    train_df.loc[val_idx, 'is_val'] = True

    get_features(train_df)


if __name__ == '__main__':
    main()
