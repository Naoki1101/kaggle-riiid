import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../src')
import const

n_history_step = 5
part_dim = 7
default_array = np.zeros(n_history_step, dtype=np.int32)


def onehot_encoder(array, dim_num):
    output = np.identity(dim_num)[array]

    nan_idx = np.where(array == -1)[0]
    if len(nan_idx) > 0:
        output[nan_idx] = 0

    return output


def update_dict(history_dict, key, value):
    history_step = history_dict[key]
    history_step = np.delete(history_step, 0)
    history_step = np.append(history_step, value)
    history_dict[key] = history_step
    return history_dict


def extract_features(df):

    history_part_dict = {}
    # history_timestamp_dict = {}
    history_correctly_dict = {}

    history_part_feats = np.zeros((len(df), n_history_step, part_dim + 1), dtype=np.int32)

    for idx, row in enumerate(tqdm(df[['user_id', 'part', 'timestamp', 'answered_correctly']].values)):
        if row[3] != -1:
            onehot_encoder(history_part_dict.setdefault(row[0], default_array), part_dim)
            history_part_feats[idx, :, :-1] = onehot_encoder(history_part_dict.setdefault(row[0], default_array - 1), part_dim)
            history_part_feats[idx, :, -1] = history_correctly_dict.setdefault(row[0], default_array - 1)
            # history_part_feats[idx, :, -1] = history_timestamp_dict.setdefault(row[0], default_array)

            history_part_dict = update_dict(history_part_dict, key=row[0], value=row[1])
            history_correctly_dict = update_dict(history_part_dict, key=row[0], value=row[3])
            # history_timestamp_dict = update_dict(history_timestamp_dict, key=row[0], value=row[2])

    return history_part_feats


def get_features(df):
    history_feats = extract_features(df)
    np.save('../data/processed/history_feats.npy', history_feats)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    question_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv')
    question_df

    q2p = dict(question_df[['question_id', 'part']].values)
    train_df['part'] = train_df['content_id'].map(q2p) - 1
    train_df['part'] = train_df['part'].fillna(-1).astype(np.int32)

    get_features(train_df)


if __name__ == '__main__':
    main()
