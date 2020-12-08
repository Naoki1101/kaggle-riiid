import sys
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
import const
from factory import get_fold
from utils import DataHandler
from feature_utils import save_features, TargetEncoding

cfg = edict({
    'name': 'KFold',
    'params': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 0,
    },
    'split': {
        'y': 'user_id',
        'groups': None
    },
    'weight': 'average'
})

dh = DataHandler()


def get_features(df):
    features_df = pd.DataFrame()

    df['content_id_and_attempt_c'] = df['content_id'].astype(str) + '_' + df['attempt_c'].astype(str)

    fold_df = get_fold(cfg, df)

    for col in ['content_id', 'tag', 'part', 'content_id_and_attempt_c']:
        te = TargetEncoding(fold_df)
        features_df[f'te_{col}_by_answered_correctly'] = te.fit_transform(df[col], df['answered_correctly'])
        dh.save(f'../data/processed/te_{col}_by_answered_correctly.pkl', te.encoder)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    lectures_df = pd.read_csv(const.INPUT_DATA_DIR / 'lectures.csv')
    lectures_df.rename(columns={'lecture_id': 'content_id'}, inplace=True)

    attempt_c = pd.read_feather('../features/attempt_c_train.feather')['attempt_c'].values
    train_df['attempt_c'] = np.where(attempt_c <= 3, attempt_c, 4)

    train_df = pd.merge(train_df, lectures_df, on='content_id', how='left')

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
