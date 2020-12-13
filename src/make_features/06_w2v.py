import sys
import random
import numpy as np
import pandas as pd
from gensim.models import word2vec

sys.path.append('../src')
import const
from utils import DataHandler
from feature_utils import save_features

w2v_params = {
    "size": 32,
    "iter": 1_000,
    "window": 128,
    "seed": 2020,
    "min_count": 1,
    "workers": -1
}

dh = DataHandler()


def get_features(df, question_df):
    features_df = pd.DataFrame()

    question_df['tags_list'] = question_df['tags'].apply(lambda x: str(x).split(' '))

    tag_list = question_df['tags_list'].tolist()
    shuffled_tag_list = []
    for n in tag_list:
        random.shuffle(n)
        shuffled_tag_list.append(n)

    tag_list += shuffled_tag_list

    n_components = w2v_params['size']
    model = word2vec.Word2Vec(tag_list, **w2v_params)
    vocab_keys = list(model.wv.vocab.keys())
    w2v_array = np.zeros((len(vocab_keys), n_components))

    for i, v in enumerate(vocab_keys):
        w2v_array[i, :] = model.wv[v]
    vocab_vectors_df = pd.DataFrame(w2v_array, columns=[f'v{i}' for i in range(n_components)])
    vocab_vectors_df['tag'] = vocab_keys

    tag_exploded_df = question_df[['question_id', 'tags_list']].explode('tags_list')
    tag_exploded_df.columns = ['question_id', 'tag']

    for i in range(n_components):
        le = dict(vocab_vectors_df[['tag', f'v{i}']].values)
        tag_exploded_df[f'v{i}'] = tag_exploded_df['tag'].map(le)

    aggs = {'mean', 'median', 'std', 'max', 'min'}
    for i in range(n_components):
        agg_df = dict(tag_exploded_df.groupby('question_id')[f'v{i}'].agg(aggs))

        for col in aggs:
            le = dict(agg_df[col])
            features_df[f'tag_w2v_{col}'] = df['content_id'].map(le)

            dh.save(f'../data/processed/tag_w2v_{col}_encoder.pkl', le)

    return features_df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)
    question_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv')

    train_features_df = get_features(train_df, question_df)
    save_features(train_features_df, data_type='train')


if __name__ == '__main__':
    main()
