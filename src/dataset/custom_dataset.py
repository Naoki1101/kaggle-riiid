# import gc
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('../src')
import const
from utils import DataHandler

dh = DataHandler()


# https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates
class CustomTrainDataset(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        self.user_ids = []
        for user_id in samples.index:
            q, qa, qp = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qp_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qp[-seq_len:] = qp_

        target_id = q[1:]
        label = qa[1:]
        part = qp[1:]

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        feat = {
            'x': torch.LongTensor(x),
            'target_id': torch.LongTensor(target_id),
            'part': torch.LongTensor(part)
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.user_ids = [x for x in df['user_id'].unique()]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_id = row['user_id']
        target_id = row['content_id']

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_, qt_ = self.samples[user_id]

            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill

        questions = np.append(q[2:], [target_id])

        feat = {
            'x': torch.LongTensor(x),
            'target_id': torch.LongTensor(questions),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(qa[2:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT
class CustomTrainDataset2(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset2, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        self.user_ids = []
        for user_id in samples.index:
            q, _, _, _, _ = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, _, _, qp_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }
        # print(user_id)
        # print(feat)

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset2(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset2, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        # self.user_ids = [x for x in df['user_id'].unique()]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq/row_{int(row_id)}.pkl')

        feat = {
            'in_ex': torch.LongTensor(seq_list[0]),
            'in_cat': torch.LongTensor(seq_list[1]),
            'in_de': torch.LongTensor(seq_list[2]),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[2][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v2
class CustomTrainDataset3(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset3, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        self.user_ids = []
        for user_id in samples.index:
            q, _, _, _, _ = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qc_, _, qp_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qc = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qc[:] = qc_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qc[:] = qa_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qc[-seq_len:] = qc_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qc[-seq_len:] = qc_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        task_container_id = qc[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_tc': torch.LongTensor(task_container_id),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset3(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset3, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        # self.user_ids = [x for x in df['user_id'].unique()]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq2/row_{int(row_id)}.pkl')

        feat = {
            'in_ex': torch.LongTensor(seq_list[0]),
            'in_tc': torch.LongTensor(seq_list[1]),
            'in_cat': torch.LongTensor(seq_list[3]),
            'in_de': torch.LongTensor(seq_list[4]),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[2][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
class CustomMlpDataset(Dataset):
    def __init__(self, df, cfg=None):
        super(CustomMlpDataset, self).__init__()

        # self.cfg = cfg

        self.feats = df.values
        self.cols = df.columns.tolist()
        self.is_train = False

        if const.TARGET_COLS[0] in self.cols:
            self.labels = df[const.TARGET_COLS].values
            target_col_idx = self.cols.index(const.TARGET_COLS[0])
            self.feats = np.delete(self.feats, target_col_idx, axis=1)
            self.is_train = True

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])

        if self.is_train:
            label = torch.tensor(self.labels[idx]).float()
            return feat, label
        else:
            return feat


class CustomMlpDataset2(Dataset):
    def __init__(self, df, cfg=None):
        super(CustomMlpDataset2, self).__init__()

        self.contents = df[['content_id']].values
        self.is_train = False

        drop_cols = ['content_id']

        if const.TARGET_COLS[0] in df.columns:
            self.labels = df[const.TARGET_COLS].values
            self.is_train = True
            drop_cols += const.TARGET_COLS
        else:
            self.feats = df.drop(['content_id'], axis=1).values

        self.feats = df.drop(drop_cols, axis=1).values

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat = {
            'x': torch.FloatTensor(self.feats[idx]),
            'content': torch.LongTensor(self.contents[idx])
        }

        if self.is_train:
            label = torch.FloatTensor(self.labels[idx])
            return feat, label
        else:
            return feat
