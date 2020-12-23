import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features


def add_user_feats(df, user_part_dict):
    user_part_count10 = np.zeros(len(df))
    user_part_count20 = np.zeros(len(df))
    user_part_count30 = np.zeros(len(df))
    default_array = np.zeros(30) - 1

    for idx, row in enumerate(tqdm(df[['user_id', 'part']].values)):
        user_id = row[0]
        part = row[1]

        user_hist_part_array = user_part_dict.setdefault(user_id, default_array.copy())
        user_part_count10[idx] = len(np.where(user_hist_part_array[-10:] == part)[0])
        user_part_count20[idx] = len(np.where(user_hist_part_array[-20:] == part)[0])
        user_part_count30[idx] = len(np.where(user_hist_part_array[-30:] == part)[0])

        new_list = list(user_part_dict[user_id])
        new_list.pop(0)
        new_list.append(part)
        user_part_dict[user_id] = np.array(new_list)

    user_feats_df = pd.DataFrame({
        'user_part_count10': user_part_count10,
        'user_part_count20': user_part_count20,
        'user_part_count30': user_part_count30,
    })

    return user_feats_df


def get_features(df):
    features_df = pd.DataFrame()

    user_part_dict = {}
    features_df = add_user_feats(df, user_part_dict)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    questions_df = pd.read_csv('../data/input/questions.csv')
    q2p = dict(questions_df[['question_id', 'part']].values)
    train_df['part'] = train_df['content_id'].map(q2p)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
