import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    df['part2_flg'] = (df['part'] == 2).astype(float)
    df['part5_flg'] = (df['part'] == 5).astype(float)
    df['one'] = 1

    part2_count_array = np.zeros(len(df))
    part5_count_array = np.zeros(len(df))
    user_count = np.zeros(len(df))

    user_gp = df.groupby('user_id')
    for user_id, user_df in tqdm(user_gp, total=len(user_gp)):
        user_idx = user_df.index.values

        part2_count_array[user_idx] = user_df['part2_flg'].cumsum()
        part5_count_array[user_idx] = user_df['part5_flg'].cumsum()
        user_count[user_idx] = user_df['one'].cumsum()

    features_df['count_part2'] = part2_count_array
    features_df['count_part5'] = part5_count_array
    features_df['user_count'] = user_count
    features_df['rate_part2'] = features_df['count_part2'] / features_df['user_count']
    features_df['rate_part5'] = features_df['count_part5'] / features_df['user_count']
    features_df['rate_part2_5'] = (features_df['count_part2'] + features_df['count_part5']) / features_df['user_count']

    features_df = features_df.drop(['user_count'], axis=1)

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
