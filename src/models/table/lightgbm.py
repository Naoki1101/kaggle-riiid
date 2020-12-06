import sys
import logging

import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import optuna.integration.lightgbm as lgb_tuner

sys.path.append('../src')
import factory
from .base import Model


class LightGBM(Model):
    def __init__(self, cfg):
        super(LightGBM, self).__init__(cfg.params, cfg.task_type)
        self.best_params = {}
        self.tuning_history = []
        self.cfg = cfg

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None):
        if cat_features is not None:
            tr_x[cat_features] = tr_x[cat_features].astype('category')
            va_x[cat_features] = va_x[cat_features].astype('category')

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        fobj = factory.get_lgb_objective(self.cfg.fobj)
        feval = factory.get_lgb_feval(self.cfg.feval)

        callbacks = [self._log_evaluation(period=100)]

        if self.cfg.task_type in ['binary', 'classification', 'regression']:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=self.cfg.num_boost_round,
                                   valid_sets=[lgb_train, lgb_eval],
                                   verbose_eval=self.cfg.verbose_eval,
                                   early_stopping_rounds=self.cfg.early_stopping_rounds,
                                   callbacks=callbacks,
                                   fobj=fobj,
                                   feval=feval)
        elif self.cfg.task_type == 'optuna':
            self.model = lgb_tuner.train(self.params,
                                         lgb_train,
                                         num_boost_round=self.cfg.num_boost_round,
                                         valid_sets=[lgb_train, lgb_eval],
                                         best_params=self.best_params,
                                         tuning_history=self.tuning_history,
                                         verbose_eval=self.cfg.verbose_eval,
                                         early_stopping_rounds=self.cfg.early_stopping_rounds,
                                         callbacks=callbacks)

            print('Number of finished trials: {}'.format(len(self.tuning_history)))
            print('Best params:', self.best_params)
            print('  Params: ')
            for key, value in self.best_params.items():
                print('    {}: {}'.format(key, value))

    def predict(self, te_x, cat_features=None):
        if cat_features is not None:
            te_x[cat_features] = te_x[cat_features].astype('category')
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def extract_importances(self):
        return self.model.feature_importance(importance_type=self.cfg.imp_type)

    def _log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
        logger = logging.getLogger('main')

        def _callback(env):
            if period > 0 and env.evaluation_result_list \
                    and (env.iteration + 1) % period == 0:
                result = '\t'.join([
                    _format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))

        _callback.order = 10
        return _callback


def lightgbm(cfg):
    model = LightGBM(cfg)
    return model
