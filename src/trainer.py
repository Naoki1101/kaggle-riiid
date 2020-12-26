import gc
import sys
import time
import logging
import matplotlib.pyplot as plt

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from fastprogress import master_bar, progress_bar
import torch
from torch.autograd import detect_anomaly
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../src')
import factory
from utils import DataHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cat_features = self.cfg.data.features.cat_features
        self.oof = None
        self.raw_preds = None
        self.target_columns = []
        self.weights = []
        self.models = []
        self.scores = []
        self.feature_importance_df = pd.DataFrame(columns=['feature', 'importance'])
        self.dh = DataHandler()
        self.thresh_hold = 0.5

    def train(self, train_df: pd.DataFrame, target_df: pd.DataFrame, fold_df: pd.DataFrame):
        self.target_columns = target_df.columns.tolist()
        self.oof = np.zeros(len(train_df))

        for fold_, col in enumerate(fold_df.columns):
            print(f'\n========================== FOLD {fold_ + 1} / {len(fold_df.columns)} ... ==========================\n')
            logging.debug(f'\n========================== FOLD {fold_ + 1} / {len(fold_df.columns)} ... ==========================\n')

            self._train_fold(train_df, target_df, fold_df[col])

        cv = np.mean(self.scores)

        print('\n\n===================================\n')
        print(f'CV: {cv:.6f}')
        print('\n===================================\n\n')
        logging.debug('\n\n===================================\n')
        logging.debug(f'CV: {cv:.6f}')
        logging.debug('\n===================================\n\n')

        return cv

    def _train_fold(self, train_df, target_df, fold):
        tr_x, va_x = train_df[fold == 0], train_df[fold > 0]
        tr_y, va_y = target_df[fold == 0], target_df[fold > 0]
        weight = fold.max()
        self.weights.append(weight)

        model = factory.get_model(self.cfg.model)
        model.fit(tr_x, tr_y, va_x, va_y, self.cat_features)
        va_pred = model.predict(va_x, self.cat_features)

        if self.cfg.data.target.reconvert_type:
            va_y = getattr(np, self.cfg.data.target.reconvert_type)(va_y)
            va_pred = getattr(np, self.cfg.data.target.reconvert_type)(va_pred)
            va_pred = np.where(va_pred >= 0, va_pred, 0)

        self.models.append(model)
        self.oof[va_x.index] = va_pred.copy()

        score = factory.get_metrics(self.cfg.common.metrics.name)(va_y, va_pred)
        print(f'\n{self.cfg.common.metrics.name}: {score:.6f}\n')
        self.scores.append(score)

        if self.cfg.model.name in ['lightgbm', 'catboost', 'xgboost']:
            importance_fold_df = pd.DataFrame()
            fold_importance = model.extract_importances()
            importance_fold_df['feature'] = train_df.columns
            importance_fold_df['importance'] = fold_importance
            self.feature_importance_df = pd.concat([self.feature_importance_df, importance_fold_df], axis=0)

    def predict(self, test_df):
        preds = np.zeros(len(test_df))
        for fold_, model in enumerate(self.models):
            pred = model.predict(test_df, self.cat_features)
            if self.cfg.data.target.reconvert_type:
                pred = getattr(np, self.cfg.data.target.reconvert_type)(pred)
                pred = np.where(pred >= 0, pred, 0)
            preds += pred.copy() * self.weights[fold_]
        self.raw_preds = preds.copy()

        return preds

    def save(self, run_name):
        log_dir = Path(f'../logs/{run_name}')
        self.dh.save(log_dir / 'oof.npy', self.oof)
        self.dh.save(log_dir / 'raw_preds.npy', self.raw_preds)
        self.dh.save(log_dir / 'importance.csv', self.feature_importance_df)
        self.dh.save(log_dir / 'model_weight.pkl', self.models)


class NNTrainer:
    def __init__(self, run_name, fold_df, cfg):
        self.run_name = run_name
        self.cfg = cfg
        self.fold_df = fold_df
        self.n_splits = len(fold_df.columns)
        self.oof = None
        self.raw_preds = None

    def train(self, train_df, target_df):
        oof = np.zeros((len(train_df), self.cfg.model.n_classes))
        cv = 0

        for fold_, col in enumerate(self.fold_df.columns):
            print(f'\n========================== FOLD {fold_ + 1} / {self.n_splits} ... ==========================\n')
            logging.debug(f'\n========================== FOLD {fold_ + 1} / {self.n_splits} ... ==========================\n')

            trn_x, val_x = train_df[self.fold_df[col] == 0], train_df[self.fold_df[col] > 0]
            val_y = target_df[self.fold_df[col] > 0].values

            if 'transformer' in self.cfg.model.backbone:
                usecols = ['user_id', 'content_id', 'timestamp', 'prior_question_elapsed_time', 'part', 'answered_correctly']
                group = (trn_x[usecols]
                         .groupby('user_id')
                         .apply(lambda r: (r['content_id'].values,
                                           r['answered_correctly'].values,
                                           r['timestamp'].values,
                                           r['prior_question_elapsed_time'].values,
                                           r['part'].values)))

                train_loader = factory.get_transformer_dataloader(samples=group, df=None, cfg=self.cfg.data.train)
                valid_loader = factory.get_transformer_dataloader(samples=group, df=val_x, cfg=self.cfg.data.valid)
            else:
                train_loader = factory.get_dataloader(trn_x, self.cfg.data.train)
                valid_loader = factory.get_dataloader(val_x, self.cfg.data.valid)

            model = factory.get_nn_model(self.cfg).to(device)

            criterion = factory.get_loss(self.cfg)
            optimizer = factory.get_optim(self.cfg, model.parameters())
            scheduler = factory.get_scheduler(self.cfg, optimizer)

            best_epoch = -1
            best_val_score = -np.inf
            mb = master_bar(range(self.cfg.model.epochs))

            train_loss_list = []
            val_loss_list = []
            val_score_list = []

            for epoch in mb:
                start_time = time.time()

                with detect_anomaly():
                    model, avg_loss = self._train_epoch(model, train_loader, criterion, optimizer, mb)

                valid_preds, avg_val_loss = self._val_epoch(model, valid_loader, criterion)

                val_score = factory.get_metrics(self.cfg.common.metrics.name)(val_y, valid_preds)

                train_loss_list.append(avg_loss)
                val_loss_list.append(avg_val_loss)
                val_score_list.append(val_score)

                if self.cfg.scheduler.name != 'ReduceLROnPlateau':
                    scheduler.step()
                elif self.cfg.scheduler.name == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)

                elapsed = time.time() - start_time
                mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f} val_score: {val_score:.6f} time: {elapsed:.0f}s')
                logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f} val_score: {val_score:.6f} time: {elapsed:.0f}s')

                if val_score > best_val_score:
                    best_epoch = epoch + 1
                    best_val_score = val_score
                    best_valid_preds = valid_preds
                    if self.cfg.model.multi_gpu:
                        best_model = model.module.state_dict()
                    else:
                        best_model = model.state_dict()

            oof[val_x.index, :] = best_valid_preds
            cv += best_val_score * self.fold_df[col].max()

            torch.save(best_model, f'../logs/{self.run_name}/weight_best_{fold_}.pt')
            # self._save_loss_png(train_loss_list, val_loss_list, val_score_list, fold_)

            print(f'\nEpoch {best_epoch} - val_score: {best_val_score:.6f}')
            logging.debug(f'\nEpoch {best_epoch} - val_score: {best_val_score:.6f}')

        print('\n\n===================================\n')
        print(f'CV: {cv:.6f}')
        logging.debug(f'\n\nCV: {cv:.6f}')
        print('\n===================================\n\n')

        self.oof = oof

        return cv

    def predict(self, test_df):
        all_preds = np.zeros((len(test_df), self.cfg.model.n_classes * self.n_splits))
        result_preds = np.zeros((len(test_df), self.cfg.model.n_classes))

        for fold_num, col in enumerate(self.fold_df.columns):
            test_df.to_csv(f'../notebooks/test_{fold_num}.csv', index=False)
            preds = self._predict_fold(fold_num, test_df)
            all_preds[:, fold_num * self.cfg.model.n_classes: (fold_num + 1) * self.cfg.model.n_classes] = preds

        for i in range(self.cfg.model.n_classes):
            preds_col_idx = [i + self.cfg.model.n_classes * j for j in range(self.n_splits)]
            result_preds[:, i] = np.mean(all_preds[:, preds_col_idx], axis=1)

        self.raw_preds = result_preds

        return result_preds

    def save(self):
        log_dir = Path(f'../logs/{self.run_name}')
        np.save(log_dir / 'oof.npy', self.oof)
        np.save(log_dir / 'raw_preds.npy', self.raw_preds)

    def _train_epoch(self, model, train_loader, criterion, optimizer, mb):
        model.train()
        avg_loss = 0.

        for feats, targets in progress_bar(train_loader, parent=mb):
            # print(feats['in_ex'])
            # print(feats['in_cat'])
            # print(feats['in_de'])
            if type(feats) == dict:
                for k, v in feats.items():
                    feats[k] = v.to(device)
            else:
                feats = feats.to(device)
            targets = targets.to(device)

            preds = model(feats)

            # print(preds)
            # print(targets)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        del feats, targets; gc.collect()
        return model, avg_loss

    def _val_epoch(self, model, valid_loader, criterion):
        model.eval()
        valid_preds = np.zeros((len(valid_loader.dataset), self.cfg.model.n_classes))

        avg_val_loss = 0.
        valid_batch_size = valid_loader.batch_size

        with torch.no_grad():
            for i, (feats, targets) in enumerate(valid_loader):
                if type(feats) == dict:
                    for k, v in feats.items():
                        feats[k] = v.to(device)
                else:
                    feats = feats.to(device)
                targets = targets.to(device)

                preds = model(feats)

                loss = criterion(preds, targets)

                if 'transformer' in self.cfg.model.backbone:
                    preds = preds[:, -1]
                valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size, :] = preds.sigmoid().cpu().detach().numpy().reshape(-1, 1)
                avg_val_loss += loss.item() / len(valid_loader)

        return valid_preds, avg_val_loss

    def _predict_fold(self, fold_num, test_df):
        test_loader = factory.get_dataloader(test_df, self.cfg.data.test)

        model = factory.get_nn_model(self.cfg, is_train=False).to(device)
        model.load_state_dict(torch.load(f'../logs/{self.run_name}/weight_best_{fold_num}.pt'))

        all_preds = []

        model.eval()
        with torch.no_grad():
            for i, feats in enumerate(test_loader):
                if type(feats) == dict:
                    for k, v in feats.items():
                        feats[k] = v.to(device)
                else:
                    feats = feats.to(device)

                preds, _ = model(feats)
                preds = preds.sigmoid().cpu().detach().numpy()

                all_preds.append(preds)

        return np.concatenate(all_preds)

    def _save_loss_png(self, train_loss_list, val_loss_list, val_score_list, fold_num):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        ax1.plot(range(len(train_loss_list)), train_loss_list, color='blue', linestyle='-', label='train_loss')
        ax1.plot(range(len(val_loss_list)), val_loss_list, color='green', linestyle='-', label='val_loss')
        ax1.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.set_title(f'Training and validation {self.cfg.loss.name}')
        ax1.grid()

        ax2.plot(range(len(val_score_list)), val_score_list, color='blue', linestyle='-', label='val_score')
        ax2.legend()
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('score')
        ax2.set_title('Training and validation score')
        ax2.grid()

        plt.savefig(f'../logs/{self.run_name}/learning_curve_{fold_num}.png')


def opt_ensemble_weight(cfg, y_true, oof_list, metric):
    def objective(trial):
        p_list = [0 for i in range(len(oof_list))]
        for i in range(len(oof_list) - 1):
            p_list[i] = trial.suggest_discrete_uniform(f'p{i}', 0.0, 1.0 - sum(p_list), 0.01)
        p_list[-1] = round(1 - sum(p_list[:-1]), 2)

        y_pred = np.zeros(len(y_true))
        for i in range(len(oof_list)):
            y_pred += oof_list[i] * p_list[i]

        return metric(y_true, y_pred)

    study = optuna.create_study(direction=cfg.opt_params.direction)
    if hasattr(cfg.opt_params, 'n_trials'):
        study.optimize(objective, n_trials=cfg.opt_params.n_trials)
    elif hasattr(cfg.opt_params, 'timeout'):
        study.optimize(objective, timeout=cfg.opt_params.timeout)
    else:
        raise(NotImplementedError)
    best_params = list(study.best_params.values())
    best_weight = best_params + [round(1 - sum(best_params), 2)]

    return best_weight
