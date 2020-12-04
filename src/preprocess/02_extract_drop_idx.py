import sys

import numpy as np
import pandas as pd

sys.path.append('../src')
import const


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    lecture_idx = train_df[train_df['content_type_id'] == 1].index
    np.save('../pickle/lecture_idx.npy', lecture_idx.values)


if __name__ == '__main__':
    main()
