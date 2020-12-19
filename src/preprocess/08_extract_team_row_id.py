import numpy as np
import pandas as pd


def main():
    train_10m_df = pd.read_feather('../data/team/X_tra_wo_lec_10M.feather')
    train_20m_df = pd.read_feather('../data/team/X_tra_wo_lec_20M.feather')
    val_df = pd.read_feather('../data/team/X_val_wo_lec.feather')

    np.save('../data/team/train_10m_row_id.npy', train_10m_df['row_id'].values)
    np.save('../data/team/train_20m_row_id.npy', train_20m_df['row_id'].values)
    np.save('../data/team/valid_row_id.npy', val_df['row_id'].values)


if __name__ == '__main__':
    main()
