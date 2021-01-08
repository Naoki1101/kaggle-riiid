import gc
import argparse
import datetime
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import factory
import const
from trainer import NNTrainer
from utils import (DataHandler, Notificator, Timer, seed_everything,
                   Git, Kaggle)

warnings.filterwarnings('ignore')

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/compe.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

dh = DataHandler()
cfg = dh.load(options.common)
cfg.update(dh.load(f'../configs/exp/{options.model}.yml'))

notify_params = dh.load(options.notify)

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / 'train.log', level=logging.DEBUG)

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        if cfg.common.debug:
            train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE, nrows=5_000_000)
        else:
            train_df = pd.read_csv(const.INPUT_DATA_DIR / 'train.csv', dtype=const.DTYPE)

    with t.timer('preprocess'):
        questions_df = pd.read_csv(const.INPUT_DATA_DIR / 'questions.csv')
        q2p = dict(questions_df[['question_id', 'part']].values)
        train_df['part'] = train_df['content_id'].map(q2p)

        train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(float)

        te_content_df = pd.read_feather('../features/te_content_id_by_answered_correctly_train.feather')
        avg_u_target_df = pd.read_feather('../features/answered_correctly_avg_u_train.feather')

        if cfg.common.debug:
            te_content_df = te_content_df.iloc[:5_000_000]
            avg_u_target_df = avg_u_target_df.iloc[:5_000_000]

        train_df['te_content_id_by_answered_correctly'] = te_content_df['te_content_id_by_answered_correctly']
        train_df['answered_correctly_avg_u'] = avg_u_target_df['answered_correctly_avg_u']

    with t.timer('make folds'):
        valid_idx = np.load('../data/processed/cv1_valid.npy')
        if cfg.common.debug:
            valid_idx = valid_idx[np.where(valid_idx < len(train_df))]

        fold_df = pd.DataFrame(index=range(len(train_df)))
        fold_df['fold_0'] = 0
        fold_df.loc[valid_idx, 'fold_0'] += 1

    with t.timer('drop index'):
        if cfg.common.drop:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            if cfg.common.debug:
                drop_idx = drop_idx[np.where(drop_idx < len(train_df))]
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

        train_df['step'] = train_df.groupby('user_id').cumcount() // 200
        train_df['user_step_id'] = train_df['user_id'].astype(str) + '__' + train_df['step'].astype(str)

    with t.timer('train model'):
        trainer = NNTrainer(run_name, fold_df, cfg)
        cv = trainer.train(train_df, target_df=train_df[const.TARGET_COLS[0]])
        trainer.save()

        run_name_cv = f'{run_name}_{cv:.4f}'
        logger_path.rename(f'../logs/{run_name_cv}')
        logging.disable(logging.FATAL)

    with t.timer('kaggle api'):
        kaggle = Kaggle(cfg, run_name_cv)
        if cfg.common.kaggle.data:
            kaggle.create_dataset()
        if cfg.common.kaggle.notebook:
            kaggle.push_notebook()

    with t.timer('notify'):
        process_minutes = t.get_processing_time()
        notificator = Notificator(run_name=run_name_cv,
                                  model_name=cfg.model.backbone,
                                  cv=round(cv, 4),
                                  process_time=round(process_minutes, 2),
                                  comment=comment,
                                  params=notify_params)
        notificator.send_line()
        notificator.send_notion()
        # notificator.send_slack()

    with t.timer('git'):
        git = Git(run_name=run_name_cv)
        git.push()
        git.save_hash()


if __name__ == '__main__':
    main()
