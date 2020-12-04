import sys
import pandas as pd
from cuml.manifold import TSNE

sys.path.append('../src')
import const
from feature_utils import save_features, get_sparse_matrix

N_COMP = 2


def get_features(df):
    features_df = pd.DataFrame()

    sparse_matrix, index_values, columns_values = get_sparse_matrix(df,
                                                                    index='content_id',
                                                                    columns='user_id',
                                                                    values='simple_count')

    content_matrix = sparse_matrix.toarray()
    tsne = TSNE(n_components=N_COMP, random_state=0)
    tsne_array = tsne.fit_transform(content_matrix)

    for i in range(N_COMP):
        features_df[f'content_id_tsne_{i}'] = tsne_array[:, i]

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    train_df = train_df.iloc[:5_000_000]

    train_features_df = get_features(train_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
