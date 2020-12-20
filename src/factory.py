import gc
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../src')
import loss
import const
import models
import metrics
import validation
from utils import reduce_mem_usage, DataHandler
from dataset import custom_dataset
from models.nn.custom_model import CustomModel

dh = DataHandler()


def get_fold(cfg, df):
    df_ = df.copy()

    for col in [cfg.split.y, cfg.split.groups]:
        if col and col not in df_.columns:
            if cfg.name != 'MultilabelStratifiedKFold':
                feat = dh.load(f'../features/{col}.feather')
                df_[col] = feat[col]

            elif cfg.name == 'MultilabelStratifiedKFold':
                col = getattr(const, col)
                for c in col:
                    feat = dh.load(f'../features/{c}.feather')
                    df_[c] = feat[c]

    fold_df = pd.DataFrame(index=range(len(df_)))

    if cfg.weight == 'average':
        weight_list = [1 / cfg.params.n_splits for i in range(cfg.params.n_splits)]
    else:
        weight_list = cfg.weight
    assert len(weight_list) == cfg.params.n_splits

    fold = getattr(validation, cfg.name)(cfg)
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(df_)):
        fold_df[f'fold_{fold_}'] = 0
        fold_df.loc[val_idx, f'fold_{fold_}'] = weight_list[fold_]

    return fold_df


def get_model(cfg):
    model = getattr(models, cfg.name)(cfg=cfg)
    return model


def get_metrics(cfg):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array, drop_idx):
    filled_array = np.zeros(len(dropped_array) + len(drop_idx))
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx] = dropped_array
    return filled_array


def get_features(features, cfg):
    dfs = []
    for f in features:
        f_path = Path(f'../features/{f}_{cfg.data_type}.feather')
        log_dir = Path(f'../logs/{f}')

        if f_path.exists():
            feat = dh.load(f_path)

        elif log_dir.exists():
            feat = get_result(log_dir, cfg, cfg.data_type)
            feat = pd.DataFrame(feat, columns=[f])

        if cfg.reduce:
            df = reduce_mem_usage(feat)
        dfs.append(feat)

    df = pd.concat(dfs, axis=1)

    del dfs; gc.collect()

    return df


def get_result(log_dir, cfg, data_type):
    # log_dir = Path(f'../logs/{log_name}')

    if data_type == 'train':
        model_preds = dh.load(log_dir / 'oof.npy')
        model_cfg = dh.load(log_dir / 'config.yml')

        if model_cfg.common.drop:
            drop_name_list = []
            for drop_name in model_cfg.common.drop:
                drop_name_list.append(drop_name)

            drop_idxs = get_drop_idx(drop_name_list)
            model_preds = fill_dropped(model_preds, drop_idxs)

    elif data_type == 'test':
        model_preds = dh.load(log_dir / 'raw_preds.npy')

    model_preds_shape = model_preds.shape
    if len(model_preds_shape) > 1:
        if model_preds_shape[1] == 1:
            model_preds = model_preds.reshape(-1)

    return model_preds


def get_target(cfg):
    target = pd.read_feather(f'../features/{cfg.name}.feather')
    if cfg.convert_type is not None:
        target = getattr(np, cfg.convert_type)(target)
    return target


def get_drop_idx(cfg):
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(f'../data/processed/{drop_name}.npy')
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx


def get_ad(cfg, train_df, test_df):
    whole_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    target = np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))])
    target_df = pd.DataFrame({f'{cfg.data.target.name}': target.astype(int)})

    fold_df = get_fold(cfg.validation, whole_df)
    if cfg.validation.single:
        col = fold_df.columns[-1]
        fold_df = fold_df[[col]]
        fold_df /= fold_df[col].max()

    return whole_df, target_df, fold_df


def get_lgb_objective(cfg):
    if cfg:
        obj = lambda x, y: getattr(loss, cfg.name)(x, y, **cfg.params)
    else:
        obj = None
    return obj


def get_lgb_feval(cfg):
    if cfg:
        feval = lambda x, y: getattr(metrics, cfg.name)(x, y, **cfg.params)
    else:
        feval = None
    return feval


def get_nn_model(cfg, is_train=True):
    model = CustomModel(cfg)

    if cfg.model.multi_gpu and is_train:
        model = nn.DataParallel(model)

    return model


def get_loss(cfg):
    if hasattr(nn, cfg.loss.name):
        loss_ = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    elif hasattr(loss, cfg.loss.name):
        loss_ = getattr(loss, cfg.loss.name)(**cfg.loss.params)
    return loss_


def get_transformer_dataloader(samples, df, cfg):
    dataset = getattr(custom_dataset, cfg.dataset_type)(samples, df, cfg)
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_dataloader(df, cfg):
    dataset = getattr(custom_dataset, cfg.dataset_type)(df, cfg)
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optimizer.name)(params=parameters, **cfg.optimizer.params)
    return optim


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg.scheduler.params,
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer,
            **cfg.scheduler.params,
        )
    return scheduler
