import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from cuml.cluster import KMeans

sys.path.append('../src')
import const
from feature_utils import save_features

N_CLUSTERS = 1_000


def create_content_class(df):
    dfs = []
    for i in range(2):
        tsne_df = pd.read_feather(f'../features/dropped___content_id_tsne_{i}_train.feather')
        tsne_df[f'dropped___content_id_tsne_{i}'].fillna(-100, inplace=True)
        tsne_df[f'dropped___content_id_tsne_{i}'] = (tsne_df[f'dropped___content_id_tsne_{i}'] - tsne_df[f'dropped___content_id_tsne_{i}'].mean()) / tsne_df[f'dropped___content_id_tsne_{i}'].std()
        dfs.append(tsne_df)

    tsne_array = pd.concat(dfs, axis=1).values
    pred = KMeans(n_clusters=N_CLUSTERS).fit_predict(tsne_array)

    df[f'content_id_class{N_CLUSTERS}'] = pred

    return df


def add_user_feats(df, prior_q_dict, attempt_c_dict, attempt_cc_dict):
    pcf = np.zeros(len(df), dtype=np.int32)
    ac = np.zeros(len(df), dtype=np.int32)

    df = create_content_class(df)

    for cnt, row in enumerate(tqdm(df[['user_id', f'content_id_class{N_CLUSTERS}', 'answered_correctly']].values)):
        if row[1] == prior_q_dict.setdefault(row[0], -1):
            pcf[cnt] = 1
        else:
            pcf[cnt] = 0

        ac[cnt] = attempt_c_dict.setdefault(row[0], {}).setdefault(row[1], 1)

        prior_q_dict[row[0]] = row[1]
        attempt_c_dict[row[0]][row[1]] += 1

    user_feats_df = pd.DataFrame({
        f'equal_prior_question_flag_class{N_CLUSTERS}': pcf,
        f'attempt_c_class{N_CLUSTERS}': ac
    })
    user_feats_df.columns = [f'{col}_class{N_CLUSTERS}' for col in user_feats_df.columns]

    return user_feats_df


def get_features(df):
    features_df = pd.DataFrame()

    prior_q_dict = {}
    attempt_c_dict = {}
    attempt_cc_dict = {}

    features_df = add_user_feats(df, prior_q_dict, attempt_c_dict, attempt_cc_dict)

    features_df.columns = [f'dropped___{col}' for col in features_df.columns]

    return features_df


def main():
    train_df = pd.read_csv('../data/processed/train_dropped.csv', dtype=const.DTYPE)

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
