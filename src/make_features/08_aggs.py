import sys
import pandas as pd

sys.path.append('../src')
import const
from feature_utils import save_features, Aggregation

s = 1e-5


def get_features(df):
    features_df = pd.DataFrame()

    features_df['part'] = df['part']
    df['user_id_and_part'] = df['user_id'].astype(str) + '_' + df['part'].astype(str)
    df['content_id_bin'] = df['content_id'].apply(lambda x: x // 30)

    for key_col in ['user_id_and_part', 'content_id', 'content_id_bin']:
        col = 'prior_question_elapsed_time'

        agg = Aggregation(by=key_col,
                          columns=col,
                          aggs={'mean', 'median', 'std', 'max', 'min'})
        agg_df = agg.fit_transform(df)
        agg.agg_df.to_csv(f'../data/processed/agg_{col}_by_{key_col}_df.csv')

        for agg_col in agg_df.columns:
            features_df[f'{agg_col}_by_{key_col}'] = agg_df[agg_col]

        features_df[f'{col}_div_mean_by_{key_col}'] = df[col] / (agg_df[f'{col}_mean'] + s)
        features_df[f'{col}_div_median_by_{key_col}'] = df[col] / (agg_df[f'{col}_median'] + s)

        features_df[f'{col}_diff_mean_by_{key_col}'] = df[col] - agg_df[f'{col}_mean']
        features_df[f'{col}_diff_median_by_{key_col}'] = df[col] - agg_df[f'{col}_median']

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    question_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv')
    question_df.rename(columns={'question_id': 'content_id'}, inplace=True)

    train_df = pd.merge(train_df, question_df, on='content_id', how='left')

    usecols = [
        'user_id', 'content_id', 'part', 'prior_question_elapsed_time'
    ]
    train_features_df = get_features(train_df[usecols])
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
