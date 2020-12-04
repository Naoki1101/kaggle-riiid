import sys
import pandas as pd

sys.path.append('../src')
import const


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    for cols in [const.ID_COLS, const.TARGET_COLS]:
        for col in cols:
            df = train_df[[col]]

            df.to_feather(f'../features/{col}.feather')


if __name__ == '__main__':
    main()
