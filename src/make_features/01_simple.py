import sys
import pandas as pd

sys.path.append('../src')
import const
from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    for col in const.NUMERICAL_COLS + const.CATEGORICAL_COLS:
        if col == 'prior_question_had_explanation':
            features_df[col] = df[col].astype(float)
        else:
            features_df[col] = df[col]

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
