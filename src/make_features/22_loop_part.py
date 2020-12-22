import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features

SEQ_LEN = 100
s = 1e-3


def add_user_feats(df, user_part_dict):
    user_part_c_count = np.zeros(len(df))
    default_array = np.array([np.nan, 0])

    for idx, row in enumerate(tqdm(df[['user_id', 'part']].values)):
        user_id = row[0]
        part = row[1]

        prev_part, counter = user_part_dict.setdefault(user_id, default_array.copy())
        if part == prev_part:
            user_part_c_count[idx] = counter + 1
            user_part_dict[user_id][1] = counter + 1
        else:
            user_part_c_count[idx] = 1
            user_part_dict[user_id][0] = part
            user_part_dict[user_id][1] = 1

    user_feats_df = pd.DataFrame({
        'user_part_continous_count': user_part_c_count,
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
