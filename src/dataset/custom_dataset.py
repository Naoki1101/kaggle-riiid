# import gc
import sys
import random

import numpy as np
import pandas as pd
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

        if self.max_seq == 101:
            seq_list = dh.load(f'../data/seq/row_{int(row_id)}.pkl')
        elif self.max_seq == 121:
            seq_list = dh.load(f'../data/seq5/row_{int(row_id)}.pkl')
        elif self.max_seq == 151:
            seq_list = dh.load(f'../data/seq6/row_{int(row_id)}.pkl')

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
        q_, qa_, qt_, _, qp_ = self.samples[user_id]
        qt_ = qt_ / 60_000   # ms -> m
        # qe_ = qe_ // 1_000   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        # qe = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qt[:] = qt_[start: end]
                # qe[:] = qe_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                # qe[:] = qe_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                # qe[-seq_len:] = qe_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qt[-seq_len:] = qt_
                # qe[-seq_len:] = qe_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 0, difftime)
        difftime = np.log1p(difftime)

        # over10m_idx = np.where((difftime > 10) & (difftime <= 1_440))[0]
        # over1d_idx = np.where(difftime > 1_440)[0]
        # if len(over10m_idx) > 0:
        #     difftime[over10m_idx] = (difftime[over10m_idx] // 10) * 10
        # if len(over1d_idx) > 0:
        #     difftime[over1d_idx] = 1_440

        # difftime = difftime / 1_440.0
        # difftime = np.where(difftime > 1_008, 1_008, difftime) / 1_008

        # prior_elapsed = qe[1:].copy()
        # prior_elapsed = np.where(np.isnan(prior_elapsed), 0, prior_elapsed)
        # prior_elapsed = np.where(difftime > 300, 300, prior_elapsed)

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            # 'in_el': torch.LongTensor(prior_elapsed),
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

        seq_list = dh.load(f'../data/seq3/row_{int(row_id)}.pkl')

        difftime = seq_list[1] / 60_000   # ms -> m
        difftime = np.where(difftime < 0, 0, difftime)
        difftime = np.log1p(difftime)

        # over10m_idx = np.where((difftime > 10) & (difftime <= 1_440))[0]
        # over1d_idx = np.where(difftime > 1_440)[0]
        # if len(over10m_idx) > 0:
        #     difftime[over10m_idx] = (difftime[over10m_idx] // 10) * 10
        # if len(over1d_idx) > 0:
        #     difftime[over1d_idx] = 1_440

        # difftime = difftime / 1_440
        # difftime = np.where(difftime > 1_008, 1_008, difftime) / 1_008

        # prior_elapsed = seq_list[2]
        # prior_elapsed = np.where(np.isnan(prior_elapsed), 0, prior_elapsed)
        # prior_elapsed = np.where(prior_elapsed > 300, 300, prior_elapsed)

        feat = {
            'in_ex': torch.LongTensor(seq_list[0]),
            'in_dt': torch.FloatTensor(difftime),
            # 'in_el': torch.LongTensor(prior_elapsed),
            'in_cat': torch.LongTensor(seq_list[3]),
            'in_de': torch.LongTensor(seq_list[4]),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[4][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v3
class CustomTrainDataset4(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset4, self).__init__()
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
        q_, qa_, qt_, qe_, qp_ = self.samples[user_id]
        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qt[:] = qt_[start: end]
                qe[:] = qe_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                qe[:] = qe_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                qe[-seq_len:] = qe_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qt[-seq_len:] = qt_
                qe[-seq_len:] = qe_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset4(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset4, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq3/row_{int(row_id)}.pkl')

        difftime = seq_list[1] / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = seq_list[2] / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        feat = {
            'in_ex': torch.LongTensor(seq_list[0]),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_cat': torch.LongTensor(seq_list[3]),
            'in_de': torch.LongTensor(seq_list[4]),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[4][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v4
class CustomTrainDataset5(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset5, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

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
        q_, qa_, qt_, qe_, qp_ = self.samples[user_id]
        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qt[:] = qt_[start: end]
                qe[:] = qe_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                qe[:] = qe_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                qe[-seq_len:] = qe_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qt[-seq_len:] = qt_
                qe[-seq_len:] = qe_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset5(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset5, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq3/row_{int(row_id)}.pkl')

        difftime = seq_list[1] / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = seq_list[2] / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        feat = {
            'in_ex': torch.LongTensor(seq_list[0]),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_cat': torch.LongTensor(seq_list[3]),
            'in_de': torch.LongTensor(seq_list[4]),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[4][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v5
class CustomTrainDataset6(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset6, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

        self.user_ids = []
        for user_id in samples.index:
            q, _, _, _, _, _ = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qt_, qe_, qx_, qp_ = self.samples[user_id]
        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qt[:] = qt_[start: end]
                qe[:] = qe_[start: end]
                qx[:] = qx_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                qe[:] = qe_[-self.max_seq:]
                qx[:] = qx_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                qe[-seq_len:] = qe_[0: seq_len]
                qx[-seq_len:] = qx_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qt[-seq_len:] = qt_
                qe[-seq_len:] = qe_
                qx[-seq_len:] = qx_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()

        exp = np.where(np.isnan(exp), 0, exp)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset6(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset6, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq8/row_{int(row_id)}.pkl')

        difftime = seq_list[1] / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = seq_list[2] / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        content_id = seq_list[0]
        exp = seq_list[3]
        part = seq_list[4]
        target = seq_list[5]

        exp = np.where(np.isnan(exp), 0, exp)

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(seq_list[5][1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v6
class CustomTrainDataset7(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset7, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

        self.user_ids = []
        for user_id in samples.index:
            q, _, _, _, _, _, _ = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qtsk_, qt_, qe_, qx_, qp_ = self.samples[user_id]
        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qtsk = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qtsk[:] = qtsk_[start: end]
                qt[:] = qt_[start: end]
                qe[:] = qe_[start: end]
                qx[:] = qx_[start: end]
                qp[:] = qp_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qtsk[:] = qtsk_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                qe[:] = qe_[-self.max_seq:]
                qx[:] = qx_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qtsk[-seq_len:] = qtsk_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                qe[-seq_len:] = qe_[0: seq_len]
                qx[-seq_len:] = qx_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qtsk[-seq_len:] = qtsk_
                qt[-seq_len:] = qt_
                qe[-seq_len:] = qe_
                qx[-seq_len:] = qx_
                qp[-seq_len:] = qp_

        target_id = q[1:].copy()
        task_id = qtsk[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()

        exp = np.where(np.isnan(exp), 0, exp)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_task': torch.FloatTensor(np.log1p(task_id)),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset7(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset7, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq9/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[2]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[3]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        content_id = seq_list[0]
        task_id = seq_list[1]
        exp = seq_list[4]
        part = seq_list[5]
        target = seq_list[6]

        exp = np.where(np.isnan(exp), 0, exp)

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_task': torch.FloatTensor(np.log1p(task_id)),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v7
class CustomTrainDataset8(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset8, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

        self.user_ids = []
        for user_id in samples.index:
            q = samples[user_id][0]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qt_, qe_, qx_, qp_, qte_, qu_ = self.samples[user_id]
        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qte = np.zeros(self.max_seq, dtype=float)
        qu = np.zeros(self.max_seq, dtype=float)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start: end]
                qa[:] = qa_[start: end]
                qt[:] = qt_[start: end]
                qe[:] = qe_[start: end]
                qx[:] = qx_[start: end]
                qp[:] = qp_[start: end]
                qte[:] = qte_[start: end]
                qu[:] = qu_[start: end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
                qt[:] = qt_[-self.max_seq:]
                qe[:] = qe_[-self.max_seq:]
                qx[:] = qx_[-self.max_seq:]
                qp[:] = qp_[-self.max_seq:]
                qte[:] = qte_[-self.max_seq:]
                qu[:] = qu_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0: seq_len]
                qa[-seq_len:] = qa_[0: seq_len]
                qt[-seq_len:] = qt_[0: seq_len]
                qe[-seq_len:] = qe_[0: seq_len]
                qx[-seq_len:] = qx_[0: seq_len]
                qp[-seq_len:] = qp_[0: seq_len]
                qte[-seq_len:] = qte_[0: seq_len]
                qu[-seq_len:] = qu_[0: seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_
                qt[-seq_len:] = qt_
                qe[-seq_len:] = qe_
                qx[-seq_len:] = qx_
                qp[-seq_len:] = qp_
                qte[-seq_len:] = qte_
                qu[-seq_len:] = qu_

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()
        te_content_id = qte[1:].copy()
        avg_u_target = qu[1:].copy()

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
            'num_feat': torch.FloatTensor(num_feat),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset8(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset8, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq10/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[1]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[2]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        content_id = seq_list[0]
        exp = seq_list[3]
        part = seq_list[4]
        target = seq_list[5]
        te_content_id = seq_list[6]
        avg_u_target = seq_list[7]

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
            'num_feat': torch.FloatTensor(num_feat),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v7, user_step_id
class CustomTrainDataset9(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset9, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

        user_ids = []
        for user_id in samples.index:
            q = samples[user_id][0]
            if len(q) < 2:
                continue
            user_ids.append(user_id)

        self.user_step_ids = df[df['user_id'].isin(user_ids)]['user_step_id'].unique()

    def __len__(self):
        return len(self.user_step_ids)

    def __getitem__(self, index):
        user_step_id = self.user_step_ids[index]
        user_id, step_id = list(map(int, user_step_id.split('__')))

        q_, qa_, qt_, qe_, qx_, qp_, qte_, qu_ = self.samples[user_id]
        step_start, step_end = step_id * 200, (step_id + 1) * 200
        if step_id > 0 and len(q_[step_start: step_end]) < 200:
            step_start = (step_id - 1) * 200

        q_ = q_[step_start: step_end]
        qa_ = qa_[step_start: step_end]
        qt_ = qt_[step_start: step_end]
        qe_ = qe_[step_start: step_end]
        qx_ = qx_[step_start: step_end]
        qp_ = qp_[step_start: step_end]
        qte_ = qte_[step_start: step_end]
        qu_ = qu_[step_start: step_end]

        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qte = np.zeros(self.max_seq, dtype=float)
        qu = np.zeros(self.max_seq, dtype=float)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            start = random.randint(0, (seq_len - self.max_seq))
            end = start + self.max_seq
            q[:] = q_[start: end]
            qa[:] = qa_[start: end]
            qt[:] = qt_[start: end]
            qe[:] = qe_[start: end]
            qx[:] = qx_[start: end]
            qp[:] = qp_[start: end]
            qte[:] = qte_[start: end]
            qu[:] = qu_[start: end]
        else:
            start = 0
            end = random.randint(2, seq_len)
            seq_len = end - start
            q[-seq_len:] = q_[0: seq_len]
            qa[-seq_len:] = qa_[0: seq_len]
            qt[-seq_len:] = qt_[0: seq_len]
            qe[-seq_len:] = qe_[0: seq_len]
            qx[-seq_len:] = qx_[0: seq_len]
            qp[-seq_len:] = qp_[0: seq_len]
            qte[-seq_len:] = qte_[0: seq_len]
            qu[-seq_len:] = qu_[0: seq_len]

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()
        te_content_id = qte[1:].copy()
        avg_u_target = qu[1:].copy()

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
            'num_feat': torch.FloatTensor(num_feat),
        }

        label = torch.FloatTensor(label)

        return feat, label


# CustomTestDataset8と同じ
class CustomTestDataset9(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset9, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq10/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[1]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[2]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        content_id = seq_list[0]
        exp = seq_list[3]
        part = seq_list[4]
        target = seq_list[5]
        te_content_id = seq_list[6]
        avg_u_target = seq_list[7]

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
            'num_feat': torch.FloatTensor(num_feat),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v7, user_step_id(step_size=200)
class CustomTrainDataset10(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset10, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

        user_ids = []
        for user_id in samples.index:
            q = samples[user_id][0]
            if len(q) < 2:
                continue
            user_ids.append(user_id)

        self.user_step_ids = df[df['user_id'].isin(user_ids)]['user_step_id'].unique()

    def __len__(self):
        return len(self.user_step_ids)

    def __getitem__(self, index):
        user_step_id = self.user_step_ids[index]
        user_id, step_id = list(map(int, user_step_id.split('__')))

        q_, qa_, qt_, qe_, qx_, qp_, qte_, _ = self.samples[user_id]
        step_start, step_end = step_id * 200, (step_id + 1) * 200
        if step_id > 0 and len(q_[step_start: step_end]) < 200:
            step_start = (step_id - 1) * 200

        q_ = q_[step_start: step_end]
        qa_ = qa_[step_start: step_end]
        qt_ = qt_[step_start: step_end]
        qe_ = qe_[step_start: step_end]
        qx_ = qx_[step_start: step_end]
        qp_ = qp_[step_start: step_end]
        qte_ = qte_[step_start: step_end]

        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qte = np.zeros(self.max_seq, dtype=float)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int)

        if seq_len >= self.max_seq:
            start = random.randint(0, (seq_len - self.max_seq))
            end = start + self.max_seq
            q[:] = q_[start: end]
            qa[:] = qa_[start: end]
            qt[:] = qt_[start: end]
            qe[:] = qe_[start: end]
            qx[:] = qx_[start: end]
            qp[:] = qp_[start: end]
            qte[:] = qte_[start: end]
        else:
            start = 0
            end = random.randint(2, seq_len)
            seq_len = end - start
            q[-seq_len:] = q_[0: seq_len]
            qa[-seq_len:] = qa_[0: seq_len]
            qt[-seq_len:] = qt_[0: seq_len]
            qe[-seq_len:] = qe_[0: seq_len]
            qx[-seq_len:] = qx_[0: seq_len]
            qp[-seq_len:] = qp_[0: seq_len]
            qte[-seq_len:] = qte_[0: seq_len]

        target_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()
        te_content_id = qte[1:].copy()

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(target_id > 0)[0][0]
        ac_latest = ac[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        prior_cid = 0
        for idx, cid in enumerate(target_id):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(target_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
            'num_feat': torch.FloatTensor(num_feat),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset10(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset10, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [0] * (6 - len(x)) + x)
        self.q2tg = dict(questions_df[['question_id', 'tag_list']].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq10/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[1]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[2]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg = np.zeros((self.max_seq - 1, 6))
        prior_cid = 0
        for idx, cid in enumerate(seq_list[0]):
            if cid == 0 and prior_cid == 0:
                qtg[idx, :] = np.zeros(6) + 188
            else:
                qtg[idx, :] = self.q2tg[cid]

            prior_cid = cid

        content_id = seq_list[0]
        exp = seq_list[3]
        part = seq_list[4]
        target = seq_list[5]
        te_content_id = seq_list[6]
        avg_u_target = seq_list[7]

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(content_id > 0)[0][0]
        ac_latest = target[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
            'num_feat': torch.FloatTensor(num_feat),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v7, faster...?
class CustomTrainDataset11(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset11, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [188] * (6 - len(x)) + x)
        self.tag_array = np.array(list(questions_df['tag_list'].values))

        user_ids = []
        for user_id in samples.index:
            q = samples[user_id][0]
            if len(q) < 2:
                continue
            user_ids.append(user_id)

        self.user_step_ids = df[df['user_id'].isin(user_ids)]['user_step_id'].unique()

    def __len__(self):
        return len(self.user_step_ids)

    def __getitem__(self, index):
        user_step_id = self.user_step_ids[index]
        user_id, step_id = list(map(int, user_step_id.split('__')))

        q_, qa_, qt_, qe_, qx_, qp_, qte_, _ = self.samples[user_id]
        step_start, step_end = step_id * 200, (step_id + 1) * 200
        if step_id > 0 and len(q_[step_start: step_end]) < 200:
            step_start = (step_id - 1) * 200

        q_ = q_[step_start: step_end]
        qa_ = qa_[step_start: step_end]
        qt_ = qt_[step_start: step_end]
        qe_ = qe_[step_start: step_end]
        qx_ = qx_[step_start: step_end]
        qp_ = qp_[step_start: step_end]
        qte_ = qte_[step_start: step_end]

        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qx = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qte = np.zeros(self.max_seq, dtype=float)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int) + 188

        if seq_len >= self.max_seq:
            start = random.randint(0, (seq_len - self.max_seq))
            end = start + self.max_seq
            q[:] = q_[start: end]
            qa[:] = qa_[start: end]
            qt[:] = qt_[start: end]
            qe[:] = qe_[start: end]
            qx[:] = qx_[start: end]
            qp[:] = qp_[start: end]
            qte[:] = qte_[start: end]
        else:
            start = 0
            end = random.randint(2, seq_len)
            seq_len = end - start
            q[-seq_len:] = q_[0: seq_len]
            qa[-seq_len:] = qa_[0: seq_len]
            qt[-seq_len:] = qt_[0: seq_len]
            qe[-seq_len:] = qe_[0: seq_len]
            qx[-seq_len:] = qx_[0: seq_len]
            qp[-seq_len:] = qp_[0: seq_len]
            qte[-seq_len:] = qte_[0: seq_len]

        content_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        exp = qx[1:].copy()
        ac = qa[:-1].copy()
        te_content_id = qte[1:].copy()

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(content_id > 0)[0][0]
        ac_latest = ac[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg[start_idx:, :] = self.tag_array[content_id[start_idx:]]

        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
            'num_feat': torch.FloatTensor(num_feat),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset11(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset11, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [188] * (6 - len(x)) + x)
        self.tag_array = np.array(list(questions_df['tag_list'].values))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq10/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[1]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[2]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        content_id = seq_list[0]
        exp = seq_list[3]
        part = seq_list[4]
        target = seq_list[5]
        te_content_id = seq_list[6]
        avg_u_target = seq_list[7]

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(content_id > 0)[0][0]
        ac_latest = target[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        exp = np.where(np.isnan(exp), 0, exp)
        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        qtg = np.zeros((self.max_seq - 1, 6)) + 188
        qtg[start_idx:, :] = self.tag_array[content_id[start_idx:]]

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_exp': torch.LongTensor(exp),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
            'num_feat': torch.FloatTensor(num_feat),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
            label = torch.FloatTensor(label)
            return feat, label
        else:
            return feat


# =================================================================================
# SAINT v7, faster...?
class CustomTrainDataset12(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset12, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [188] * (6 - len(x)) + x)
        self.tag_array = np.array(list(questions_df['tag_list'].values))

        user_ids = []
        for user_id in samples.index:
            q = samples[user_id][0]
            if len(q) < 2:
                continue
            user_ids.append(user_id)

        self.user_step_ids = df[df['user_id'].isin(user_ids)]['user_step_id'].unique()

    def __len__(self):
        return len(self.user_step_ids)

    def __getitem__(self, index):
        user_step_id = self.user_step_ids[index]
        user_id, step_id = list(map(int, user_step_id.split('__')))

        q_, qa_, qt_, qe_, _, qp_, qte_, _ = self.samples[user_id]
        step_start, step_end = step_id * 200, (step_id + 1) * 200
        if step_id > 0 and len(q_[step_start: step_end]) < 200:
            step_start = (step_id - 1) * 200

        q_ = q_[step_start: step_end]
        qa_ = qa_[step_start: step_end]
        qt_ = qt_[step_start: step_end]
        qe_ = qe_[step_start: step_end]
        qp_ = qp_[step_start: step_end]
        qte_ = qte_[step_start: step_end]

        qt_ = qt_ / 60_000.   # ms -> m
        qe_ = qe_ / 1_000.   # ms -> s
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qt = np.zeros(self.max_seq, dtype=int)
        qe = np.zeros(self.max_seq, dtype=int)
        qp = np.zeros(self.max_seq, dtype=int)
        qte = np.zeros(self.max_seq, dtype=float)
        qtg = np.zeros((self.max_seq - 1, 6), dtype=int) + 188

        if seq_len >= self.max_seq:
            start = random.randint(0, (seq_len - self.max_seq))
            end = start + self.max_seq
            q[:] = q_[start: end]
            qa[:] = qa_[start: end]
            qt[:] = qt_[start: end]
            qe[:] = qe_[start: end]
            qp[:] = qp_[start: end]
            qte[:] = qte_[start: end]
        else:
            start = 0
            end = random.randint(2, seq_len)
            seq_len = end - start
            q[-seq_len:] = q_[0: seq_len]
            qa[-seq_len:] = qa_[0: seq_len]
            qt[-seq_len:] = qt_[0: seq_len]
            qe[-seq_len:] = qe_[0: seq_len]
            qp[-seq_len:] = qp_[0: seq_len]
            qte[-seq_len:] = qte_[0: seq_len]

        content_id = q[1:].copy()
        label = qa[1:].copy()
        part = qp[1:].copy()
        ac = qa[:-1].copy()
        te_content_id = qte[1:].copy()

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(content_id > 0)[0][0]
        ac_latest = ac[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)

        difftime = np.diff(qt.copy())
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = qe[1:].copy()
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        qtg[start_idx:, :] = self.tag_array[content_id[start_idx:]]

        num_feat = np.vstack([te_content_id, avg_u_target]).T

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(ac),
            'num_feat': torch.FloatTensor(num_feat),
        }

        label = torch.FloatTensor(label)

        return feat, label


class CustomTestDataset12(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTestDataset12, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples
        self.df = df

        questions_df = pd.read_csv('../data/input/questions.csv')
        questions_df['tags'] = questions_df['tags'].fillna(0)
        questions_df['tag_list'] = questions_df['tags'].apply(lambda tags: [int(tag) for tag in str(tags).split(' ')])
        questions_df['tag_list'] = questions_df['tag_list'].apply(lambda x: [188] * (6 - len(x)) + x)
        self.tag_array = np.array(list(questions_df['tag_list'].values))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_id = row['row_id']

        seq_list = dh.load(f'../data/seq10/row_{int(row_id)}.pkl')

        difftime = np.array(seq_list[1]) / 60_000.   # ms -> m
        difftime = np.where(difftime < 0, 300, difftime)
        difftime = np.log1p(difftime)

        prior_elapsed = np.array(seq_list[2]) / 1_000.
        prior_elapsed = np.log1p(prior_elapsed)
        prior_elapsed = np.where(np.isnan(prior_elapsed), np.log1p(21), prior_elapsed)

        content_id = seq_list[0]
        part = seq_list[4]
        target = seq_list[5]
        te_content_id = seq_list[6]
        avg_u_target = seq_list[7]

        avg_u_target = np.zeros(self.max_seq - 1, dtype=float)
        start_idx = np.where(content_id > 0)[0][0]
        ac_latest = target[start_idx:]
        avg_u_target[start_idx:] = ac_latest.cumsum() / (np.arange(len(ac_latest)) + 1)

        te_content_id = np.where(np.isnan(te_content_id), 0.625164097637492, te_content_id)   # nanmean
        avg_u_target = np.where(np.isnan(avg_u_target), 0, avg_u_target)
        num_feat = np.vstack([te_content_id, avg_u_target]).T

        qtg = np.zeros((self.max_seq - 1, 6)) + 188
        qtg[start_idx:, :] = self.tag_array[content_id[start_idx:]]

        feat = {
            'in_ex': torch.LongTensor(content_id),
            'in_dt': torch.FloatTensor(difftime),
            'in_el': torch.FloatTensor(prior_elapsed),
            'in_tag': torch.LongTensor(qtg),
            'in_cat': torch.LongTensor(part),
            'in_de': torch.LongTensor(target),
            'num_feat': torch.FloatTensor(num_feat),
        }

        if const.TARGET_COLS[0] in self.df.columns:
            label = np.append(target[1:], [row[const.TARGET_COLS[0]]])
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
