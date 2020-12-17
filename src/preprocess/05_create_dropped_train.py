import sys
import numpy as np
import pandas as pd

sys.path.append('../src')
import const


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    cv1_valid = pd.read_pickle('../folds/cv1_valid.pickle')

    duplicated_task_row_id = np.load('../data/processed/row_id_duplicated_task.npy')
    valid_row_id = cv1_valid['row_id'].values

    train_dropped_df = train_df[~train_df['row_id'].isin(duplicated_task_row_id)].reset_index(drop=True)

    train_dropped_df.to_csv('../data/processed/train_dropped.csv', index=False)

    valid_dropped_idx = train_dropped_df[train_dropped_df['row_id'].isin(valid_row_id)].index.values
    np.save('../data/processed/cv1_valid_dropped.npy', valid_dropped_idx)


if __name__ == '__main__':
    main()
