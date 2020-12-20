import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import const
import factory
from trainer import opt_ensemble_weight
from utils import (DataHandler, Kaggle, Notificator, Timer,
                   seed_everything, Git)

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

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        train_df = dh.load('../data/team/X_tra_wo_lec_20M.feather')
        val_df = dh.load('../data/team/X_val_wo_lec.feather')

        train_df['is_val'] = 0
        val_df['is_val'] = 1

        train_df = pd.concat([train_df, val_df], axis=0, sort=False, ignore_index=True)
        val_idx = train_df[train_df['is_val'] == 1].index

    # with t.timer('load data'):
    #     train_df = dh.load('../data/input/train_features.feather')
    #     train_score_df = dh.load('../data/input/train_targets_scored.feather')

    # with t.timer('preprocess'):
    #     train_df = pd.merge(train_df, train_score_df, on=const.ID_COLS, how='left')

    with t.timer('drop index'):
        drop_idx = np.array([])
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            val_df = val_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('load oof and preds'):
        oof_list = []
        # preds_list = []

        for i, log_name in enumerate(sorted(cfg.models)):
            log_dir = Path(f'../logs/{log_name}')
            model_oof = factory.get_result(log_dir, cfg, data_type='train')

            if len(drop_idx) > 0:
                model_oof = np.delete(model_oof, drop_idx, axis=0)

            oof_list.append(model_oof[val_idx].reshape(len(val_idx), -1))
            # preds_list.append(model_preds)

    with t.timer('optimize model weight'):
        metric = factory.get_metrics(cfg.common.metrics.name)

        best_weight_array = np.zeros((len(const.TARGET_COLS), len(oof_list)))
        for target_idx, target in enumerate(const.TARGET_COLS):
            best_weight = opt_ensemble_weight(cfg,
                                              val_df[target],
                                              [oof[:, target_idx] for oof in oof_list],
                                              metric)
            best_weight_array[target_idx, :] = best_weight

    with t.timer('ensemble'):
        ensemble_oof = np.zeros((len(val_df), len(const.TARGET_COLS)))
        # ensemble_preds = np.zeros((len(test_df), len(const.TARGET_COLS)))

        cv_list = []
        for target_idx, target_col in enumerate(const.TARGET_COLS):
            for model_idx, weight in enumerate(best_weight_array[target_idx]):
                ensemble_oof[:, target_idx] += oof_list[model_idx][:, target_idx] * weight
                # ensemble_preds[:, target_idx] += preds_list[model_idx][:, target_idx] * weight

            print(ensemble_oof)
            cv_list.append(metric(val_df[target_col].values, ensemble_oof[:, target_idx]))

        dh.save(f'../logs/{run_name}/oof.npy', ensemble_oof)
        # dh.save(f'../logs/{run_name}/raw_preds.npy', ensemble_preds)
        dh.save(f'../logs/{run_name}/best_weight.npy', best_weight_array)

        cv = np.mean(cv_list)
        run_name_cv = f'{run_name}_{cv:.6f}'
        logger_path.rename(f'../logs/{run_name_cv}')

        print('\n\n===================================\n')
        print(f'CV: {cv:.6f}')
        print('\n===================================\n\n')

    with t.timer('kaggle api'):
        kaggle = Kaggle(cfg, run_name_cv)
        if cfg.common.kaggle.data:
            kaggle.create_dataset()
        if cfg.common.kaggle.notebook:
            kaggle.push_notebook()

    with t.timer('notify'):
        process_minutes = t.get_processing_time()
        notificator = Notificator(run_name=run_name_cv,
                                  model_name='ensemble',
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