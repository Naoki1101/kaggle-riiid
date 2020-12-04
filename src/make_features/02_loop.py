import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../src')
import const
from feature_utils import save_features


def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict, prior_q_dict, attempt_c_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    pcf = np.zeros(len(df), dtype=np.int32)
    ac = np.zeros(len(df), dtype=np.int32)

    for cnt, row in enumerate(tqdm(df[['user_id', 'content_id', 'answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict.setdefault(row[0], 0)
        cu[cnt] = count_u_dict.setdefault(row[0], 0)
        if row[1] == prior_q_dict.setdefault(row[0], -1):
            pcf[cnt] = 1
        else:
            pcf[cnt] = 0

        ac[cnt] = attempt_c_dict.setdefault(row[0], {}).setdefault(row[1], 1)

        answered_correctly_sum_u_dict[row[0]] += row[2]
        prior_q_dict[row[0]] = row[1]
        count_u_dict[row[0]] += 1
        attempt_c_dict[row[0]][row[1]] += 1

    user_feats_df = pd.DataFrame({
        'answered_correctly_sum_u': acsu,
        'count_u': cu,
        'equal_prior_question_flag': pcf,
        'attempt_c': ac
    })
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']

    return user_feats_df


def get_features(df):
    features_df = pd.DataFrame()

    answered_correctly_sum_u_dict = {}
    count_u_dict = {}
    prior_q_dict = {}
    attempt_c_dict = {}

    features_df = add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict, prior_q_dict, attempt_c_dict)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
