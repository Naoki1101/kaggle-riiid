import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append('../src')
import const


# https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates
class CustomTrainDataset(Dataset):
    def __init__(self, samples, df, cfg=None):
        super(CustomTrainDataset, self).__init__()
        self.max_seq = cfg.params.max_seq
        self.n_skill = cfg.params.n_skill
        self.samples = samples

        self.user_ids = []
        for user_id in samples.index:
            q, qa = samples[user_id]
            if len(q) < 2:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start:end]
                qa[:] = qa_[start:end]
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
        else:
            if random.random() > 0.1:
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0:seq_len]
                qa[-seq_len:] = qa_[0:seq_len]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        feat = {
            'x': torch.LongTensor(x),
            'target_id': torch.LongTensor(target_id)
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
            q_, qa_ = self.samples[user_id]

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
