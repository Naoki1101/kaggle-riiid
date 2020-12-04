import sys
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
import const
from factory import get_fold
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


def get_features(df):
    features_df = pd.DataFrame()

    fold_df = get_fold(cfg, df)

    for col in ['content_id', 'tag', 'part', 'type_of']:
        te = TargetEncoding(fold_df)
        features_df[f'te_{col}_by_answered_correctly'] = te.fit_transform(df[col], df['answered_correctly'])

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    lectures_df = pd.read_csv(const.INPUT_DATA_DIR / 'lectures.csv')
    lectures_df.rename(columns={'lecture_id': 'content_id'}, inplace=True)

    train_df = pd.merge(train_df, lectures_df, on='content_id', how='left')

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
