import argparse
import datetime
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import const
import factory
from trainer import Trainer
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

features_params = dh.load(f'../configs/feature/{cfg.data.features.name}.yml')
features = features_params.features

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
if cfg.model.task_type != 'optuna':
    run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'
else:
    run_name = f'{model_name}_optuna_{now:%Y%m%d%H%M%S}'

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
    dh.save(logger_path / 'features.yml', features_params)

    with t.timer('load data'):
        train_x = factory.get_features(features, cfg.data.loader.train)
        train_y = factory.get_target(cfg.data.target)

    with t.timer('add oof'):
        if cfg.data.features.oof.name is not None:
            oof, preds = factory.get_result(cfg.data.features.oof.name, cfg)

            for i in range(oof.shape[1]):
                oof_col_name = f'oof_{const.TARGET_COLS[i]}'
                train_x[oof_col_name] = oof[:, i]
                features.append(oof_col_name)

    with t.timer('make folds'):
        valid_idx = np.load('../data/processed/cv1_valid.npy')

        fold_df = pd.DataFrame(index=range(len(train_x)))
        fold_df['fold_0'] = 0
        fold_df.loc[valid_idx, 'fold_0'] += 1

    with t.timer('drop index'):
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True).iloc[:5_000_000]
            train_y = train_y.drop(drop_idx, axis=0).reset_index(drop=True).iloc[:5_000_000]
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('train and predict'):
        trainer = Trainer(cfg)
        cv = trainer.train(train_df=train_x,
                           target_df=train_y,
                           fold_df=fold_df)
        trainer.save(run_name)

        run_name_cv = f'{run_name}_{cv:.3f}'
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
                                  model_name=cfg.model.name,
                                  cv=round(cv, 4),
                                  process_time=round(process_minutes, 2),
                                  comment=comment,
                                  params=notify_params)
        notificator.send_line()
        notificator.send_notion()
        notificator.send_slack()

    with t.timer('git'):
        git = Git(run_name=run_name_cv)
        git.push()
        git.save_hash()


if __name__ == '__main__':
    main()
