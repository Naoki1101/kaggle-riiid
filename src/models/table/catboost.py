import shutil
import logging
from abc import ABCMeta, abstractmethod

from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from pathlib import Path
import pandas as pd

from .base import Model


class _BaseCB(Model, metaclass=ABCMeta):
    def __init__(self, cfg):
        super(_BaseCB, self).__init__(cfg.params, cfg.task_type)
        self.cfg = cfg
        self.cb = None

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):
        if cat_features is not None:
            tr_x[cat_features] = tr_x[cat_features].astype('category')
            va_x[cat_features] = va_x[cat_features].astype('category')

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        self.model = self.cb.fit(cb_train,
                                 eval_set=cb_valid,
                                 use_best_model=True,
                                 verbose_eval=self.cfg.verbose_eval,
                                 early_stopping_rounds=self.cfg.early_stopping_rounds,
                                 plot=False)

        self._log_evaluation(period=self.cfg.verbose_eval)

    @abstractmethod
    def predict(self, te_x, cat_features=None):
        pass

    def extract_importances(self, features=None):
        return self.model.feature_importances_

    def _log_evaluation(self, period=1, level=logging.DEBUG):
        info_dir = Path('./catboost_info')
        learn_df = pd.read_csv(info_dir / 'learn_error.tsv', sep='\t')
        test_df = pd.read_csv(info_dir / 'learn_error.tsv', sep='\t')

        logger = logging.getLogger('main')
        for iteration in range(len(learn_df)):
            if period > 0 and learn_df is not None and (iteration + 1) % period == 0:
                metrics = self.cfg.params.eval_metric
                train_loss = learn_df.iloc[iteration, 1]
                val_loss = test_df.iloc[iteration, 1]
                result = f"train-{metrics}: {train_loss:.6f}\ttvalid-{metrics}: {val_loss:.6f}"
                logger.log(level, '[{}]\t{}'.format(iteration + 1, result))

        shutil.rmtree(info_dir)


class CBBinaryClassifier(_BaseCB):
    def __init__(self, cfg):
        super(CBBinaryClassifier, self).__init__(cfg)
        self.cb = CatBoostClassifier(**cfg.params)

    def predict(self, te_x, cat_features=None):
        if cat_features is not None:
            te_x[cat_features] = te_x[cat_features].astype('category')
        return self.model.predict_proba(te_x)[:, 1]


class CBClassifier(_BaseCB):
    def __init__(self, cfg):
        super(CBClassifier, self).__init__(cfg)
        self.cb = CatBoostClassifier(**cfg.params)

    def predict(self, te_x, cat_features=None):
        if cat_features is not None:
            te_x[cat_features] = te_x[cat_features].astype('category')
        return self.model.predict_proba(te_x)[:, 1]


class CBRegressor(_BaseCB):
    def __init__(self, cfg):
        super(CBRegressor, self).__init__(cfg)
        self.cb = CatBoostRegressor(**cfg.params)

    def predict(self, te_x, cat_features=None):
        if cat_features is not None:
            te_x[cat_features] = te_x[cat_features].astype('category')
        return self.model.predict(te_x)


def catboost(cfg):
    if cfg.task_type == 'binary':
        return CBBinaryClassifier(cfg)
    elif cfg.task_type == 'classification':
        return CBClassifier(cfg)
    elif cfg.task_type == 'regression':
        return CBRegressor(cfg)
    else:
        raise(NotImplementedError)
