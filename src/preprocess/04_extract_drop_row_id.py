import sys
import numpy as np
import pandas as pd

sys.path.append('../src')
import const


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    valid_idx = np.load('../data/processed/cv1_valid.npy')
    train_df = train_df.drop(valid_idx, axis=0).reset_index(drop=True)

    train_df['user_idxxxtask_container_id'] = train_df['user_id'].astype(str) + '_' + train_df['task_container_id'].astype(str)
    train_df_sorted = train_df.sort_values(by=['user_id', 'timestamp', 'task_container_id', 'answered_correctly'],
                                           ascending=[True, True, True, False]).reset_index(drop=True)

    row_id_duplicated_task = train_df_sorted[train_df_sorted['user_idxxxtask_container_id'].duplicated()]['row_id'].values
    np.save('../data/processed/row_id_duplicated_task.npy', row_id_duplicated_task)


if __name__ == '__main__':
    main()
