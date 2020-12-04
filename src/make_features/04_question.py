import sys
import pandas as pd

sys.path.append('../src')
import const
from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    features_df['part'] = df['part']

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    question_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv', dtype=const.DTYPE)
    question_df.rename(columns={'question_id': 'content_id'}, inplace=True)

    train_df = pd.merge(train_df, question_df, on='content_id', how='left')

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
