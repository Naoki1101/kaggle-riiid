import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../src')
import const


def extract_features(df):

    ans_correctly_dict = {}
    ans_incorrectly_dict = {}
    ans_correctly_part_dict = {}
    ans_incorrectly_part_dict = {}

    ans_correctly = np.zeros(len(df), dtype=np.int32)
    ans_incorrectly = np.zeros(len(df), dtype=np.int32)
    ans_correctly_part = np.zeros(len(df), dtype=np.int32)
    ans_incorrectly_part = np.zeros(len(df), dtype=np.int32)

    for idx, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'part']].values)):
        if row[1] != -1:
            ans_correctly[idx] = ans_correctly_dict.setdefault(row[0], 0)
            ans_incorrectly[idx] = ans_incorrectly_dict.setdefault(row[0], 0)
            ans_correctly_part[idx] = ans_correctly_part_dict.setdefault(row[0], {}).setdefault(row[2], 0)
            ans_incorrectly_part[idx] = ans_incorrectly_part_dict.setdefault(row[0], {}).setdefault(row[2], 0)

            ans_correctly_dict[row[0]] += row[1]
            ans_incorrectly_dict[row[0]] += row[1]
            ans_correctly_part_dict[row[0]][row[2]] += row[1]
            ans_incorrectly_part_dict[row[0]][row[2]] += row[1]

    current_feats = np.vstack([
        np.log1p(ans_correctly),
        np.log1p(ans_incorrectly),
        np.log1p(ans_correctly_part),
        np.log1p(ans_incorrectly_part),
        np.log1p(df['te_content_id_by_answered_correctly'].values),
    ]).T

    return current_feats


def get_features(df):
    te_feats = pd.read_feather('../features/te_content_id_by_answered_correctly_train.feather')
    df['te_content_id_by_answered_correctly'] = te_feats['te_content_id_by_answered_correctly']
    current_feats = extract_features(df)
    np.save('../data/processed/current_feats.npy', current_feats)


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    question_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv')

    q2p = dict(question_df[['question_id', 'part']].values)
    train_df['part'] = train_df['content_id'].map(q2p).astype(str)

    get_features(train_df)


if __name__ == '__main__':
    main()
