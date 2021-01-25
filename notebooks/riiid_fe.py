
import os
# import sys
# import gc
# import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from bitarray import bitarray
# import pickle5
from tqdm import tqdm

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import multinomial

# from sklearn.metrics import roc_auc_score


class riiidFE:
    def __init__(self, model_parames):
        self.user_corr_cnt_dict = defaultdict(int)
        self.user_cnt_dict = defaultdict(int)

        # self.user_corr_cnt_in_session_dict = defaultdict(int)
        self.user_corr_cnt_in_session_short_dict = defaultdict(int)
        # self.user_corr_cnt_in_session_long_dict = defaultdict(int)

        # self.user_cnt_in_session_dict = defaultdict(int)
        self.user_cnt_in_session_short_dict = defaultdict(int)
        # self.user_cnt_in_session_long_dict = defaultdict(int)

        # self.user_session_cnt_dict = defaultdict(int)
        self.user_session_short_cnt_dict = defaultdict(int)
        # self.user_session_long_cnt_dict = defaultdict(int)

        # self.user_prev_session_ts_dict = defaultdict(int)
        self.user_prev_session_short_ts_dict = defaultdict(int)
        # self.user_prev_session_long_ts_dict = defaultdict(int)

        self.question_u_dict = {}
        # self.question_corr_u_dict = {}
        self.part_u_dict = {}
        self.part_corr_u_dict = {}
        self.tag_u_dict = {}
        self.tag_corr_u_dict = {}

        self.lecture_cnt_u_dict = defaultdict(int)

        self.hadexp_sum_u_dict = defaultdict(int)
        self.hadexp_cnt_u_dict = defaultdict(int)

        self.user_rec_dict = {}
        self.user_prev_ques_dict = defaultdict(list)

        self.user_timestamp_dict = defaultdict(list)
        self.user_timestamp_incorr_dict = defaultdict(int)
        self.user_et_dict = defaultdict(list)

        self.user_et_sum_dict = defaultdict(int)
        # self.user_difftime_sum_dict = defaultdict(int)
        # self.user_lagtime_sum_dict = defaultdict(int)

        self.session_short_th = 600000  # 30min model_parames['session_th']
        # self.session_th = 1800000  # 30min model_parames['session_th']
        # self.session_long_th = 43200000  # 12h, model_parames['session_th']

        self.smooth = model_parames['TE_smooth']
        self.ansrec_max_len = model_parames['ansrec_max_len']
        self.timestamprec_max_len = model_parames['timestamprec_max_len']
        self.prev_question_len = model_parames['prev_question_len']

        self.loop_features = ['timestamp', 'content_type_id', 'user_id', 'content_id', 'part', 'tags', 'prior_question_had_explanation', 'prior_question_elapsed_time']
        self.use_labels = ['answered_correctly']

        # self.sequence_te_dict = {}
        # col_list = [
        #     ['prev_part_s1', 'part'],
        #     ['prev_part_s2', 'prev_part_s1', 'part'],
        #     ['prev_part_s3', 'prev_part_s2', 'prev_part_s1', 'part'],
        #     ['prev_question_id_s1', 'question_id'],
        #     ['prev_question_id_s2', 'prev_question_id_s1', 'question_id'],
        # ]
        # for col in col_list:
        #     colname = "xxx".join(col)
        #     fname = f'{colname}__{tar}_sm{smooth}'
        #     with open(f'../save/f{col[-1]}_sequence_s{len(col)-1}_te_dict_sm{smooth}.pkl', mode='rb') as f:
        #         te_dict = pickle5.load(f)
        #     self.sequence_te_dict[fname] = te_dict
        self.se_smooth = model_parames['sequence_te_smooth']
        # self.th = model_parames['sequence_te_threshold']
        # with open(f'../save/features_{FOLD_NAME}/question_id_sequence_s1_te_dict_sm{self.se_smooth}.pkl', mode='rb') as f:
        #     self.ques_sequence_s1_te_dict = pickle5.load(f)
        # with open(f'../save/features_{FOLD_NAME}/question_id_sequence_s2_te_dict_sm{self.se_smooth}.pkl', mode='rb') as f:
        #     self.ques_sequence_s2_te_dict = pickle5.load(f)
#         with open(f'../save/features_{FOLD_NAME}/part_sequence_s1_te_dict_sm{self.se_smooth}.pkl', mode='rb') as f:
#             self.part_sequence_s1_te_dict = pickle5.load(f)
#         with open(f'../save/features_{FOLD_NAME}/part_sequence_s2_te_dict_sm{self.se_smooth}.pkl', mode='rb') as f:
#             self.part_sequence_s2_te_dict = pickle5.load(f)
#         with open(f'../save/features_{FOLD_NAME}/part_sequence_s3_te_dict_sm{self.se_smooth}.pkl', mode='rb') as f:
#             self.part_sequence_s3_te_dict = pickle5.load(f)

    def set_use_features(self, use_features):
        self.use_features = use_features
        print(f'Feature Num: {len(self.use_features)}')

    def set_train_mn(self, train):
        self.train_mn = train['answered_correctly'].mean()
        self.train_wans_mn = train['weighted_answered_correctly'].mean()
        self.train_ws_mn = train['weighted_score'].mean()
        print(self.train_mn, self.train_wans_mn, self.train_ws_mn)

    def set_repeat_mn(self, repeat):
        self.repeat_mn = repeat['answered_correctly'].mean()
        self.repeat_wans_mn = repeat['weighted_answered_correctly'].mean()
        self.repeat_ws_mn = repeat['weighted_score'].mean()
        print(self.repeat_mn, self.repeat_wans_mn, self.repeat_ws_mn)

    def set_cat_te_dict(self, X_tra_wo_lec, question):

        col = 'part'
        te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target='answered_correctly', mn=self.train_mn)   # target_encording with target
        self.part_te_dict = {int(i): j for i, j in te_feat.values}

        col = 'content_id'
        te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target='answered_correctly', mn=self.train_mn)   # target_encording with target & 平均値埋め
        self.ques_te_dict = {int(i): j for i, j in te_feat.values}
        for cid in question['question_id'].unique():
            if cid not in self.ques_te_dict:
                self.ques_te_dict[int(cid)] = self.train_mn

        self.question2part = {i[0]: i[1] for i in question[['question_id', 'part']].values}

    def init_user_log_dict(self, uid):
        if uid not in self.question_u_dict:
            self.question_u_dict[uid] = np.zeros(13523, dtype=np.uint8)
        # if uid not in self.question_corr_u_dict:
            # self.question_corr_u_dict[uid] = np.zeros(13523, dtype=np.uint8)

        if uid not in self.part_u_dict:
            self.part_u_dict[uid] = np.zeros(8, dtype=np.uint16)
        if uid not in self.part_corr_u_dict:
            self.part_corr_u_dict[uid] = np.zeros(8, dtype=np.uint16)

        if uid not in self.tag_u_dict:
            self.tag_u_dict[uid] = np.zeros(189, dtype=np.uint16)
        if uid not in self.tag_corr_u_dict:
            self.tag_corr_u_dict[uid] = np.zeros(189, dtype=np.uint16)

        if uid not in self.user_rec_dict:
            self.user_rec_dict[uid] = bitarray(0, endian='little')
        return

    def update_user_log_dict(self, uid, cid, part, tags, ans, timestamp, et, hexp, dt, lt):
        self.user_corr_cnt_dict[uid] += ans
        self.user_cnt_dict[uid] += 1

        # self.user_corr_cnt_in_session_dict[uid] += ans
        self.user_corr_cnt_in_session_short_dict[uid] += ans   ################################################要確認！
        # self.user_corr_cnt_in_session_long_dict[uid] += ans

        # self.user_cnt_in_session_dict[uid] += 1
        self.user_cnt_in_session_short_dict[uid] += 1
        # self.user_cnt_in_session_long_dict[uid] += 1

        if len(self.user_rec_dict[uid]) == self.ansrec_max_len:
            self.user_rec_dict[uid].pop(0)
        self.user_rec_dict[uid].append(ans)
        if len(self.user_prev_ques_dict[uid]) == self.prev_question_len:
            self.user_prev_ques_dict[uid].pop(0)
        self.user_prev_ques_dict[uid].append(cid)

        if len(self.user_timestamp_dict[uid]) == self.timestamprec_max_len:
            self.user_timestamp_dict[uid].pop(0)
        self.user_timestamp_dict[uid].append(timestamp)

        if len(self.user_et_dict[uid]) == self.timestamprec_max_len:
            self.user_et_dict[uid].pop(0)
        self.user_et_dict[uid].append(et)
        self.user_et_sum_dict[uid] += et
        # self.user_difftime_sum_dict[uid] += dt
        # self.user_lagtime_sum_dict[uid] += lt

        if hexp == 1:
            self.hadexp_sum_u_dict[uid] += ans
            self.hadexp_cnt_u_dict[uid] += 1

        self.question_u_dict[uid][cid] += 1
        self.part_u_dict[uid][part] += 1
        for t in tags:
            self.tag_u_dict[uid][t] += 1

        if ans == 1:
            self.part_corr_u_dict[uid][part] += 1
            # self.question_corr_u_dict[uid][cid] += 1
            for t in tags:
                self.tag_corr_u_dict[uid][t] += 1
        else: # incorrect
            self.user_timestamp_incorr_dict[uid] = timestamp

        # if dt > self.session_th:
        #     self.user_session_cnt_dict[uid] += 1
        if dt > self.session_short_th:
            self.user_session_short_cnt_dict[uid] += 1
        # if dt > self.session_long_th:
        #     self.user_session_long_cnt_dict[uid] += 1

        return

    def dataframe_process(self, user_feats_df):
        user_feats_df['u_corr_rate'] = user_feats_df['u_corr_cnt'] / user_feats_df['u_cnt']
        user_feats_df['u_hade_corr_rate'] = user_feats_df['u_hade_corr_cnt'] / user_feats_df['u_hade_cnt']
        user_feats_df['u_hade_rate'] = user_feats_df['u_hade_cnt'] / user_feats_df['u_cnt']
        user_feats_df['u_hade_div_corr_cnt'] = user_feats_df['u_hade_cnt'] / user_feats_df['u_corr_cnt']
    
        user_feats_df['u_corr_rate_smooth'] = (user_feats_df['u_cnt'] * user_feats_df['u_corr_rate'] + self.smooth * self.train_mn) / (user_feats_df['u_cnt'] + self.smooth)

        # user_feats_df['u_hist_ques_nuni_corr_rate'] = user_feats_df['u_hist_ques_corr_nuni'] / user_feats_df['u_hist_ques_nuni']
        # user_feats_df['u_hist_ques_num_corr_rate'] = user_feats_df['u_hist_ques_corr_num'] / user_feats_df['u_hist_ques_num']

        # user_feats_df['u_hist_part_nuni_corr_rate'] = user_feats_df['u_hist_part_corr_nuni'] / user_feats_df['u_hist_part_nuni']
        user_feats_df['u_hist_part_num_corr_rate'] = user_feats_df['u_hist_part_corr_num'] / user_feats_df['u_hist_part_num']

        # user_feats_df['u_hist_tag_nuni_corr_rate'] = user_feats_df['u_hist_tag_corr_nuni'] / user_feats_df['u_hist_tag_nuni']
        user_feats_df['u_hist_tag_num_corr_rate'] = user_feats_df['u_hist_tag_corr_num'] / user_feats_df['u_hist_tag_num']

        user_feats_df['u_prev_diff_lag_time'] = user_feats_df['u_prev_difftime'] / user_feats_df['u_prev_lagtime']
        user_feats_df['u_prev_lag_diff_time'] = user_feats_df['u_prev_lagtime'] / user_feats_df['u_prev_difftime']
        return user_feats_df

    def add_user_feats(self, df, add_feat=True, update_dict=True, val=False):

        def add_user_latest_record_feat(cnt, user_records):
            if len(user_records) == 0:
                u_latest3_corr_rate[cnt] = np.nan
                u_latest5_corr_rate[cnt] = np.nan
                u_latest10_corr_rate[cnt] = np.nan
                u_latest30_corr_rate[cnt] = np.nan
                u_latest100_corr_rate[cnt] = np.nan
            else:
                # ur = [user_records[i] for i in range(len(user_records)-1, -1, -1)]
                # corr_cnt = np.cumsum(ur)
                # cnt = np.array([i for i in range(1,len(user_records)+1)])
                # corr_rates = corr_cnt / cnt
                
                # u_latest1_corr_rate[cnt] = corr_rates[0]
                # u_latest3_corr_rate[cnt] = corr_rates[:3][-1]
                # u_latest5_corr_rate[cnt] = corr_rates[:5][-1]
                # u_latest10_corr_rate[cnt] = corr_rates[:10][-1]
                # u_latest30_corr_rate[cnt] = corr_rates[:30][-1]
                # u_latest100_corr_rate[cnt] = corr_rates[-1]                
                # u_corr_rate_srope[cnt] = slope(cnt[:10], corr_rates[:10]

                u_latest1_corr_rate[cnt] = user_records[-1]
                u_latest3_corr_rate[cnt] = sum(user_records[-3:]) / len(user_records[-3:])
                u_latest5_corr_rate[cnt] = sum(user_records[-5:]) / len(user_records[-5:])
                u_latest10_corr_rate[cnt] = sum(user_records[-10:]) / len(user_records[-10:])
                u_latest30_corr_rate[cnt] = sum(user_records[-30:]) / len(user_records[-30:])
                u_latest100_corr_rate[cnt] = sum(user_records) / len(user_records)
            return

        # def add_user_ques_record_feat(cnt, uid, cid, part, user_ques_records, user_corr_ques_records, val=False):
        def add_user_ques_record_feat(cnt, uid, cid, part, user_ques_records, val=False):
            
            u_hist_ques_ansed[cnt] = bool(user_ques_records[cid])
            # u_hist_ques_corr_ansed[cnt] = bool(user_corr_ques_records[cid])
            u_hist_ques_num[cnt] = user_ques_records[cid]
            # u_hist_ques_corr_num[cnt] = user_corr_ques_records[cid]
            # u_hist_ques_nuni[cnt] = user_ques_records.astype(bool).sum()
            # u_hist_ques_corr_nuni[cnt] = user_corr_ques_records.astype(bool).sum()

            if val is True:
                if uid in self.user_prev_ques_dict:
                    user_prev_questions = self.user_prev_ques_dict[uid]
                else:
                    u_prev_part_sequence_te_s1[cnt] = self.part_te_dict[part]
                    u_prev_part_sequence_te_s2[cnt] = self.part_te_dict[part]
                    u_prev_part_sequence_te_s3[cnt] = self.part_te_dict[part]
                    # u_prev_ques_sequence_te_s1[cnt] = self.ques_te_dict[cid]
                    # u_prev_ques_sequence_te_s2[cnt] = self.ques_te_dict[cid]
                    return

                prev_ques_s1 = user_prev_questions[-1]
                prev_part_s1 = self.question2part[prev_ques_s1]

                # if (prev_ques_s1, cid) in self.ques_sequence_s1_te_dict:
                #     u_prev_ques_sequence_te_s1[cnt] = self.ques_sequence_s1_te_dict[(prev_ques_s1, cid)]
                # else:
                #     u_prev_ques_sequence_te_s1[cnt] = self.ques_te_dict[cid]

                # if (prev_part_s1, part) in self.part_sequence_s1_te_dict:
                #     u_prev_part_sequence_te_s1[cnt] = self.part_sequence_s1_te_dict[(prev_part_s1, part)]
                # else:
                #     u_prev_part_sequence_te_s1[cnt] = self.part_te_dict[part]

                len_user_prev_questions = len(user_prev_questions)

                # if len_user_prev_questions > 1:
                #     prev_ques_s2 = user_prev_questions[-2]
                #     prev_part_s2 = self.question2part[prev_ques_s2]

                    # if (prev_ques_s2, prev_ques_s1, cid) in self.ques_sequence_s2_te_dict:
                    #     u_prev_ques_sequence_te_s2[cnt] = self.ques_sequence_s2_te_dict[(prev_ques_s2, prev_ques_s1, cid)]
                    # else:
                    #     u_prev_ques_sequence_te_s2[cnt] = self.ques_te_dict[cid]

                    # if (prev_part_s2, prev_part_s1, part) in self.part_sequence_s2_te_dict:
                    #     u_prev_part_sequence_te_s2[cnt] = self.part_sequence_s2_te_dict[(prev_part_s2, prev_part_s1, part)]
                    # else:
                    #     u_prev_part_sequence_te_s2[cnt] = self.part_te_dict[part]

                    # if len_user_prev_questions > 2:
                    #     prev_ques_s3 = user_prev_questions[-3]
                    #     prev_part_s3 = self.question2part[prev_ques_s3]
                    #     if (prev_part_s3, prev_part_s2, prev_part_s1, part) in self.part_sequence_s3_te_dict:
                    #         u_prev_part_sequence_te_s3[cnt] = self.part_sequence_s3_te_dict[(prev_part_s3, prev_part_s2, prev_part_s1, part)]
                    #     else:
                    #         u_prev_part_sequence_te_s3[cnt] = self.part_te_dict[part]
                    # else:
                    #     u_prev_part_sequence_te_s3[cnt] = self.part_te_dict[part]
                    #     return
                # else:
                    # u_prev_part_sequence_te_s2[cnt] = self.part_te_dict[part]
                    # u_prev_part_sequence_te_s3[cnt] = self.part_te_dict[part]
                    # u_prev_ques_sequence_te_s2[cnt] = self.ques_te_dict[cid]
                    # return
                return

        def add_user_part_record_feat(cnt, part, user_part, user_corr_part):

            # u_hist_part_ansed[cnt] = bool(user_part[part])
            # u_hist_part_corr_ansed[cnt] = bool(user_corr_part[part])
            u_hist_part_num[cnt] = user_part[part]
            u_hist_part_corr_num[cnt] = user_corr_part[part]
            # u_hist_part_nuni[cnt] = user_part.astype(bool).sum()
            # u_hist_part_corr_nuni[cnt] = user_corr_part.astype(bool).sum()

            # u_hist_part_std[cnt] = user_part.std()
            # u_hist_part_corr_std[cnt] = user_corr_part.std()
            # u_hist_part_rank[cnt] = user_part.argsort().argsort()[part]

            u_part_1_cnt[cnt] = int(user_part[1])
            u_part_2_cnt[cnt] = int(user_part[2])
            u_part_3_cnt[cnt] = int(user_part[3])
            u_part_4_cnt[cnt] = int(user_part[4])
            u_part_5_cnt[cnt] = int(user_part[5])
            u_part_6_cnt[cnt] = int(user_part[6])
            u_part_7_cnt[cnt] = int(user_part[7])

            u_part_1_corr_rate[cnt] = user_corr_part[1] / float(user_part[1] + 1)
            u_part_2_corr_rate[cnt] = user_corr_part[2] / float(user_part[2] + 1)
            u_part_3_corr_rate[cnt] = user_corr_part[3] / float(user_part[3] + 1)
            u_part_4_corr_rate[cnt] = user_corr_part[4] / float(user_part[4] + 1)
            u_part_5_corr_rate[cnt] = user_corr_part[5] / float(user_part[5] + 1)
            u_part_6_corr_rate[cnt] = user_corr_part[6] / float(user_part[6] + 1)
            u_part_7_corr_rate[cnt] = user_corr_part[7] / float(user_part[7] + 1)
            return

        def add_user_tag_record_feat(cnt, tags, user_tag, user_corr_tag):

            tag_nums = []
            tag_corr_nums = []
            tag_ranks = []
            for t in tags:
                tag_nums.append(user_tag[t])
                tag_corr_nums.append(user_corr_tag[t])
                tag_ranks.append(user_tag.argsort().argsort()[t])
            tag_corr_rates = [tag_corr_nums[i] / tag_nums[i] if tag_nums[i] != 0 else np.nan for i in range(len(tags))]

            u_hist_tag_num[cnt] = sum(tag_nums)
            u_hist_tag_corr_num[cnt] = sum(tag_corr_nums)
            # u_hist_tag_nuni[cnt] = user_tag.astype(bool).sum()
            # u_hist_tag_corr_nuni[cnt] = user_corr_tag.astype(bool).sum()

            # u_hist_tag_max[cnt] = user_tag.max()
            # u_hist_tag_min[cnt] = user_tag.min()
            # u_hist_tag_sum[cnt] = user_tag.sum()
            # u_hist_tag_std[cnt] = user_tag.std()

            # u_hist_tag_corr_max[cnt] = user_corr_tag.max()
            # u_hist_tag_corr_min[cnt] = user_corr_tag.min()
            # u_hist_tag_corr_sum[cnt] = user_corr_tag.sum()
            # u_hist_tag_corr_std[cnt] = user_corr_tag.std()
            
            u_hist_tag_rank_max[cnt] = max(tag_ranks)
            u_hist_tag_rank_min[cnt] = min(tag_ranks)
            u_hist_tag_rank_sum[cnt] = sum(tag_ranks)

            u_hist_tag_corr_rate_max[cnt] = max(tag_corr_rates)
            u_hist_tag_corr_rate_min[cnt] = min(tag_corr_rates)
            u_hist_tag_corr_rate_mn[cnt] = sum(tag_corr_rates) / len(tag_corr_rates)
            return

        def add_user_timestamp_feat(cnt, uid, timestamp, user_timestmaps, ets, user_records):
            if len(user_timestmaps) == 0:
                u_prev_difftime[cnt] = -1
                u_prev_lagtime[cnt] = -1

                u_latest5_lagtime_max[cnt] = 0
                u_latest5_lagtime_mn[cnt] = np.nan
                u_latest5_difftime_max[cnt] = 0
                u_latest5_difftime_mn[cnt] = np.nan
                # u_latest5_et_max[cnt] = 0
                u_latest5_et_mn[cnt] = np.nan
                
                u_latest10_lagtime_max[cnt] = 0
                u_latest10_lagtime_mn[cnt] = np.nan
                u_latest10_difftime_max[cnt] = 0
                u_latest10_difftime_mn[cnt] = np.nan
                u_latest10_et_max[cnt] = 0
                u_latest10_et_mn[cnt] = np.nan

                # u_hist_difftime_sum[cnt] = 0
                # u_hist_lagtime_sum[cnt] = 0

                # u_prev_difftime_norm[cnt] = np.nan
                # u_prev_lagtime_norm[cnt] = np.nan
                # u_latest5_difftime_mn_norm[cnt] = np.nan
                u_prev_et_diff_time[cnt] = np.nan
                u_prev_et_lag_time[cnt] = np.nan

                timestamp_lag_diff_rolling10_median_each_user[cnt] = np.nan

            else:
                difftimes = [0] + [user_timestmaps[i + 1] - user_timestmaps[i] for i in range(len(user_timestmaps) - 1)]
                lagtimes = [-1, -1] + [difftimes[i + 1] - ets[i + 2] for i in range(len(difftimes) - 2)]
                ets = [i for i in ets if i != -1]  # remove '-1' flag

                u_prev_difftime[cnt] = timestamp - user_timestmaps[-1]
                if len(user_timestmaps) == 2:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = -1
                    u_prev_difftime_4[cnt] = -1
                    u_prev_difftime_5[cnt] = -1
                elif len(user_timestmaps) == 3:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = timestamp - user_timestmaps[-3]
                    u_prev_difftime_4[cnt] = -1
                    u_prev_difftime_5[cnt] = -1
                elif len(user_timestmaps) == 4:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = timestamp - user_timestmaps[-3]
                    u_prev_difftime_4[cnt] = timestamp - user_timestmaps[-4]
                    u_prev_difftime_5[cnt] = -1
                elif len(user_timestmaps) == 5:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = timestamp - user_timestmaps[-3]
                    u_prev_difftime_4[cnt] = timestamp - user_timestmaps[-4]
                    u_prev_difftime_5[cnt] = timestamp - user_timestmaps[-5]
                    timestamp_lag_div_rolling5_median_each_user[cnt] = timestamp / np.median(user_timestmaps[-5:])
                    timestamp_lag_diff_rolling5_median_each_user[cnt] = timestamp - np.median(user_timestmaps[-5:])
                elif len(user_timestmaps) == 7:
                    timestamp_lag_div_rolling7_median_each_user[cnt] = timestamp / np.median(user_timestmaps[-7:])
                    timestamp_lag_diff_rolling7_median_each_user[cnt] = timestamp - np.median(user_timestmaps[-7:])
                elif len(user_timestmaps) >= 10:
                    timestamp_lag_div_rolling10_median_each_user[cnt] = timestamp / np.median(user_timestmaps[-10:])
                    timestamp_lag_diff_rolling10_median_each_user[cnt] = timestamp - np.median(user_timestmaps[-10:])
                
                u_prev_difftime_incorr[cnt] = timestamp - self.user_timestamp_incorr_dict[uid]

                if difftimes[-1] == 0:
                    u_prev_lagtime[cnt] = -1
                else:
                    u_prev_lagtime[cnt] = difftimes[-1] - et

                if ets == []:
                    ets = [0]

                u_latest5_lagtime_max[cnt] = max(lagtimes[-5:])
                u_latest5_lagtime_mn[cnt] = sum(lagtimes[-5:]) / len(lagtimes[-5:])
                u_latest5_difftime_max[cnt] = max(difftimes[-5:])
                u_latest5_difftime_mn[cnt] = sum(difftimes[-5:]) / len(difftimes[-5:])
                # u_latest5_et_max[cnt] = max(ets[-5:])
                u_latest5_et_mn[cnt] = sum(ets[-5:]) / len(ets[-5:])
                
                u_latest10_lagtime_max[cnt] = max(lagtimes)
                u_latest10_lagtime_mn[cnt] = sum(lagtimes) / len(lagtimes)
                u_latest10_difftime_max[cnt] = max(difftimes)
                u_latest10_difftime_mn[cnt] = sum(difftimes) / len(difftimes)
                u_latest10_et_max[cnt] = max(ets)
                u_latest10_et_mn[cnt] = sum(ets) / len(ets)

                # u_hist_difftime_sum[cnt] = self.user_difftime_sum_dict[uid]
                # u_hist_lagtime_sum[cnt] = self.user_lagtime_sum_dict[uid]

                # u_prev_difftime_norm[cnt] = u_prev_difftime[cnt] / timestamp
                # u_prev_lagtime_norm[cnt] = u_prev_lagtime[cnt] / timestamp
                # u_latest5_difftime_mn_norm[cnt] = u_latest5_difftime_mn[cnt] / timestamp
                u_prev_et_diff_time[cnt] = et / u_prev_difftime[cnt]
                u_prev_et_lag_time[cnt] = et / u_prev_lagtime[cnt]
            return
        
        # ここから！
        if add_feat is True and update_dict is False:   # test 特徴量を作るとき
            use_columns = self.loop_features
        else:
            use_columns = self.loop_features + self.use_labels

        if add_feat is False and update_dict is True:   # previous_test_df使って、dictをupdateするとき

            cnt = 0
            for timestamp, ctype, uid, cid, part, tags, hexp, et, ans in tqdm(df[use_columns].values):

                # init dict
                self.init_user_log_dict(uid)   # dictに存在しないuserだったら、追加する

                if ctype == 1:   # 講義だったら...
                    self.lecture_cnt_u_dict[uid] += 1
                    if len(self.user_timestamp_dict[uid]) == self.timestamprec_max_len:
                        self.user_timestamp_dict[uid].pop(0)
                    self.user_timestamp_dict[uid].append(timestamp)
                    if len(self.user_et_dict[uid]) == self.timestamprec_max_len:
                        self.user_et_dict[uid].pop(0)
                    self.user_et_dict[uid].append(et)
                    continue

                user_timestamps = self.user_timestamp_dict[uid]
                if len(user_timestamps) >= 2:
                    difftime = timestamp - user_timestamps[-1]
                    lagtime = (user_timestamps[-1] - user_timestamps[-2]) - et   ######################################要確認！
                elif len(user_timestamps) == 1:
                    difftime = timestamp - user_timestamps[-1]
                    lagtime = -1
                else:
                    difftime = 0
                    lagtime = -1
                self.update_user_log_dict(uid, cid, part, tags, ans, timestamp, et, hexp, difftime, lagtime)
            return

        if add_feat is True:   # 学習時

            rec_num = len(df[df.content_type_id == 0])

            u_cnt = np.zeros(rec_num, dtype=np.int32)   # ユーザーごとのカウント
            u_corr_cnt = np.zeros(rec_num, dtype=np.int32)    # ユーザーごとの正答数
#             u_cnt_density = np.zeros(rec_num, dtype=np.float32)
#             u_corr_cnt_density = np.zeros(rec_num, dtype=np.float32)
            
            # u_cnt_in_session = np.zeros(rec_num, dtype=np.uint16)
            u_cnt_in_session_short = np.zeros(rec_num, dtype=np.uint16)   # ？？？
            # u_cnt_in_session_long = np.zeros(rec_num, dtype=np.uint16)

            u_hade_cnt = np.zeros(rec_num, dtype=np.int32)   # ？？？
            u_hade_corr_cnt = np.zeros(rec_num, dtype=np.int32)   # ？？？

            # u_session_cnt = np.zeros(rec_num, dtype=np.int16)
            # u_session_cnt_density = np.zeros(rec_num, dtype=np.float32)
            # u_session_change = np.zeros(rec_num, dtype=np.int8)
            u_session_short_cnt = np.zeros(rec_num, dtype=np.int16)   # ？？？
            # u_session_short_cnt_density = np.zeros(rec_num, dtype=np.float32)
            # u_session_short_change = np.zeros(rec_num, dtype=np.int8)
            # u_session_long_cnt = np.zeros(rec_num, dtype=np.int16)
            # u_session_long_cnt_density = np.zeros(rec_num, dtype=np.float32)
            # u_session_long_change = np.zeros(rec_num, dtype=np.int8)
            
            # u_cnt_in_session_density = np.zeros(rec_num, dtype=np.float32)
            u_cnt_in_session_short_density = np.zeros(rec_num, dtype=np.float32)   # ？？？
            # u_cnt_in_session_long_density = np.zeros(rec_num, dtype=np.float32)
            
            # u_corr_rate_in_session = np.zeros(rec_num, dtype=np.float32)
            u_corr_rate_in_session_short = np.zeros(rec_num, dtype=np.float32)   # ？？？
            # u_corr_rate_in_session_long = np.zeros(rec_num, dtype=np.float32)
            
            u_latest1_corr_rate = np.zeros(rec_num, dtype=np.int8)   # ？？？
            u_latest3_corr_rate = np.zeros(rec_num, dtype=np.float16)   # ？？？
            u_latest5_corr_rate = np.zeros(rec_num, dtype=np.float16)   # ？？？
            u_latest10_corr_rate = np.zeros(rec_num, dtype=np.float16)   # ？？？
            u_latest30_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest100_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ？？？
            # u_corr_rate_srope = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_ansed = np.zeros(rec_num, dtype=np.uint8)   # ？？？
            u_hist_ques_num = np.zeros(rec_num, dtype=np.uint8)   # ？？？
            # u_hist_ques_corr_ansed = np.zeros(rec_num, dtype=np.uint8)
            # u_hist_ques_corr_num = np.zeros(rec_num, dtype=np.uint8)
            # u_hist_ques_nuni = np.zeros(rec_num, dtype=np.uint8)
            # u_hist_ques_corr_nuni = np.zeros(rec_num, dtype=np.uint8)

            if val is True:
                u_prev_part_sequence_te_s1 = np.zeros(rec_num, dtype=np.float32)   # ？？？
                u_prev_part_sequence_te_s2 = np.zeros(rec_num, dtype=np.float32)   # ？？？
                u_prev_part_sequence_te_s3 = np.zeros(rec_num, dtype=np.float32)   # ？？？
                # u_prev_ques_sequence_te_s1 = np.zeros(rec_num, dtype=np.float32)
                # u_prev_ques_sequence_te_s2 = np.zeros(rec_num, dtype=np.float32)

            u_hist_part_ansed = np.zeros(rec_num, dtype=np.uint8)   # ？？？
            u_hist_part_corr_ansed = np.zeros(rec_num, dtype=np.uint8)   # ？？？
            # u_hist_part_nuni = np.zeros(rec_num, dtype=np.uint8)
            # u_hist_part_corr_nuni = np.zeros(rec_num, dtype=np.uint8)
            u_hist_part_num = np.zeros(rec_num, dtype=np.uint16)   # ？？？
            u_hist_part_corr_num = np.zeros(rec_num, dtype=np.uint16)   # ？？？
            # u_hist_part_std = np.zeros(rec_num, dtype=np.float32)
            # u_hist_part_corr_std = np.zeros(rec_num, dtype=np.float32)
            # u_hist_part_rank = np.zeros(rec_num, dtype=np.uint16)

            u_part_1_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart1の出現回数
            u_part_2_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart2の出現回数
            u_part_3_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart3の出現回数
            u_part_4_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart4の出現回数
            u_part_5_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart5の出現回数
            u_part_6_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart6の出現回数
            u_part_7_cnt = np.zeros(rec_num, dtype=np.uint16)   # ユーザーごとのpart7の出現回数

            u_part_1_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart1の正解率
            u_part_2_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart2の正解率
            u_part_3_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart3の正解率
            u_part_4_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart4の正解率
            u_part_5_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart5の正解率
            u_part_6_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart6の正解率
            u_part_7_corr_rate = np.zeros(rec_num, dtype=np.float32)   # ユーザーごとのpart7の正解率
           
            u_hist_tag_num = np.zeros(rec_num, dtype=np.int16)   # ？？？
            u_hist_tag_corr_num = np.zeros(rec_num, dtype=np.int16)   # ？？？
            # u_hist_tag_nuni = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_corr_nuni = np.zeros(rec_num, dtype=np.int16)
            
            # u_hist_tag_max = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_min = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_sum = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_std = np.zeros(rec_num, dtype=np.float32)
            
            # u_hist_tag_corr_max = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_corr_min = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_corr_sum = np.zeros(rec_num, dtype=np.int16)
            # u_hist_tag_corr_std = np.zeros(rec_num, dtype=np.float32)
            
            u_hist_tag_rank_max = np.zeros(rec_num, dtype=np.int16)   # ？？？
            u_hist_tag_rank_min = np.zeros(rec_num, dtype=np.int16)   # ？？？
            u_hist_tag_rank_sum = np.zeros(rec_num, dtype=np.uint16)   # ？？？
            
            u_hist_tag_corr_rate_max = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_hist_tag_corr_rate_min = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_hist_tag_corr_rate_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            
            # u_hist_lec = np.zeros(rec_num, dtype=np.int16)
            u_hist_lec_cnt = np.zeros(rec_num, dtype=np.int16)   # ？？？

            u_prev_difftime = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_difftime_2 = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_difftime_3 = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_difftime_4 = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_difftime_5 = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_difftime_incorr = np.zeros(rec_num, dtype=np.int64)   # ？？？
            u_prev_lagtime = np.zeros(rec_num, dtype=np.int64)   # ？？？
            # u_hist_et_sum = np.zeros(rec_num, dtype=np.int64)
            u_hist_et_sum_div_cnt = np.zeros(rec_num, dtype=np.float32)   # それまでのユーザーの平均回答時間
            u_hist_et_sum_div_corr_cnt = np.zeros(rec_num, dtype=np.float32)   # それまでのユーザーの回答時間の累積 / 正答数
            # u_hist_difftime_sum = np.zeros(rec_num, dtype=np.int64)
            # u_hist_lagtime_sum = np.zeros(rec_num, dtype=np.int64)

            u_latest5_lagtime_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest5_difftime_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest5_lagtime_max = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest5_difftime_max = np.zeros(rec_num, dtype=np.float32)   # ？？？
            # u_latest5_et_max = np.zeros(rec_num, dtype=np.float32)
            u_latest5_et_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？

            u_latest10_lagtime_max = np.zeros(rec_num, dtype=np.uint32)   # ？？？
            u_latest10_difftime_max = np.zeros(rec_num, dtype=np.uint32)   # ？？？
            u_latest10_lagtime_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest10_difftime_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest10_et_mn = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_latest10_et_max = np.zeros(rec_num, dtype=np.uint32)   # ？？？
            # u_latest10_lagtime_corr_max = np.zeros(rec_num, dtype=np.uint32)
            # u_latest10_lagtime_corr_mn = np.zeros(rec_num, dtype=np.float32)
            # u_latest10_difftime_corr_max = np.zeros(rec_num, dtype=np.uint32)
            # u_latest10_difftime_corr_mn = np.zeros(rec_num, dtype=np.float32)
            # u_latest10_et_corr_max = np.zeros(rec_num, dtype=np.uint32)
            # u_latest10_et_corr_mn = np.zeros(rec_num, dtype=np.float32)

            # u_prev_difftime_norm = np.zeros(rec_num, dtype=np.float32)
            # u_prev_lagtime_norm = np.zeros(rec_num, dtype=np.float32)
            # u_latest5_difftime_mn_norm = np.zeros(rec_num, dtype=np.float32)
            u_prev_et_diff_time = np.zeros(rec_num, dtype=np.float32)   # ？？？
            u_prev_et_lag_time = np.zeros(rec_num, dtype=np.float32)   # ？？？

            # u_ts_in_session = np.zeros(rec_num, dtype=np.int32)
            u_ts_in_session_short = np.zeros(rec_num, dtype=np.int32)   # ？？？
            # u_ts_in_session_long = np.zeros(rec_num, dtype=np.int32)

            #################################
            # NARI
            #################################
            timestamp_lag_div_rolling5_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)
            timestamp_lag_div_rolling7_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)
            timestamp_lag_div_rolling10_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)
            timestamp_lag_diff_rolling5_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)
            timestamp_lag_diff_rolling7_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)
            timestamp_lag_diff_rolling10_median_each_user = np.repeat(np.nan, rec_num).astype(np.float32)

            cnt = 0
            for row in tqdm(df[use_columns].values):

                if update_dict is True:
                    timestamp, ctype, uid, cid, part, tags, hexp, et, ans = row
                else:
                    timestamp, ctype, uid, cid, part, tags, hexp, et = row
                # timestamp_sec = timestamp / 1000

                if ctype == 1:   # 講義だったら...
                    self.lecture_cnt_u_dict[uid] += 1
                    if len(self.user_timestamp_dict[uid]) == self.timestamprec_max_len:
                        self.user_timestamp_dict[uid].pop(0)
                    self.user_timestamp_dict[uid].append(timestamp)
                    if len(self.user_et_dict[uid]) == self.timestamprec_max_len:
                        self.user_et_dict[uid].pop(0)
                    self.user_et_dict[uid].append(et)
                    continue

                # init dict
                self.init_user_log_dict(uid)   # dictにuserがなかったら初期値を代入

                # add feat
                u_corr_cnt[cnt] = self.user_corr_cnt_dict[uid]
                u_cnt[cnt] = self.user_cnt_dict[uid]

                # u_session_cnt[cnt] = self.user_session_cnt_dict[uid]
                u_session_short_cnt[cnt] = self.user_session_short_cnt_dict[uid]
                # u_session_long_cnt[cnt] = self.user_session_long_cnt_dict[uid]

                # if timestamp > 0:
                #     u_corr_cnt_density[cnt] = self.user_corr_cnt_dict[uid] / timestamp_sec
                #     u_cnt_density[cnt] = self.user_cnt_dict[uid] / timestamp_sec
                #     u_session_cnt_density[cnt] = self.user_session_cnt_dict[uid] / timestamp_sec
                #     u_session_short_cnt_density[cnt] = self.user_session_short_cnt_dict[uid] / timestamp_sec
                #     u_session_long_cnt_density[cnt] = self.user_session_long_cnt_dict[uid] / timestamp_sec

                user_records = self.user_rec_dict[uid]
                add_user_latest_record_feat(cnt, user_records)

                user_ques_records = self.question_u_dict[uid]
                # user_corr_ques_records = self.question_corr_u_dict[uid]
                # add_user_ques_record_feat(cnt, uid, cid, part, user_ques_records, user_corr_ques_records, val=val)
                add_user_ques_record_feat(cnt, uid, cid, part, user_ques_records, val=val)

                u_hade_corr_cnt[cnt] = self.hadexp_sum_u_dict[uid]
                u_hade_cnt[cnt] = self.hadexp_cnt_u_dict[uid]

                user_part = self.part_u_dict[uid]
                user_corr_part = self.part_corr_u_dict[uid]
                add_user_part_record_feat(cnt, part, user_part, user_corr_part)

                user_tag = self.tag_u_dict[uid]
                user_corr_tag = self.tag_corr_u_dict[uid]
                add_user_tag_record_feat(cnt, tags, user_tag, user_corr_tag)

                # u_hist_lec[cnt] = self.prev_lecture_u_dict[uid]
                u_hist_lec_cnt[cnt] = self.lecture_cnt_u_dict[uid]

                user_timestmaps = self.user_timestamp_dict[uid]
                ets = self.user_et_dict[uid]
                add_user_timestamp_feat(cnt, uid, timestamp, user_timestmaps, ets, user_records)
                # u_hist_et_sum[cnt] = self.user_et_sum_dict[uid]
                u_hist_et_sum_div_cnt[cnt] = self.user_et_sum_dict[uid] / (self.user_cnt_dict[uid] + 1)
                u_hist_et_sum_div_corr_cnt[cnt] = self.user_et_sum_dict[uid] / (self.user_corr_cnt_dict[uid] + 1)
                
                if u_prev_difftime[cnt] > self.session_short_th:
                    # u_session_short_change[cnt] = 1
                    self.user_prev_session_short_ts_dict[uid] = timestamp
                    self.user_corr_cnt_in_session_short_dict[uid] = 0
                    self.user_cnt_in_session_short_dict[uid] = 0
                    # if u_prev_difftime[cnt] > self.session_th:
                    #     # u_session_change[cnt] = 1
                    #     self.user_prev_session_ts_dict[uid] = timestamp
                    #     self.user_corr_cnt_in_session_dict[uid] = 0
                    #     self.user_cnt_in_session_dict[uid] = 0
                    # if u_prev_difftime[cnt] > self.session_long_th:
                    #     # u_session_long_change[cnt] = 1
                    #     self.user_prev_session_long_ts_dict[uid] = timestamp
                    #     self.user_corr_cnt_in_session_long_dict[uid] = 0
                    #     self.user_cnt_in_session_long_dict[uid] = 0

                # u_ts_in_session[cnt] = timestamp - self.user_prev_session_ts_dict[uid]
                u_ts_in_session_short[cnt] = timestamp - self.user_prev_session_short_ts_dict[uid]
                # u_ts_in_session_long[cnt] = timestamp - self.user_prev_session_long_ts_dict[uid]

                # u_cnt_in_session[cnt] = self.user_cnt_in_session_dict[uid]
                u_cnt_in_session_short[cnt] = self.user_cnt_in_session_short_dict[uid]
                # u_cnt_in_session_long[cnt] = self.user_cnt_in_session_long_dict[uid]

                # if u_ts_in_session[cnt] > 0:
                #     u_cnt_in_session_density[cnt] = u_cnt_in_session[cnt] / u_ts_in_session[cnt] * 1000
                if u_ts_in_session_short[cnt] > 0:
                    u_cnt_in_session_short_density[cnt] = u_cnt_in_session_short[cnt] / u_ts_in_session_short[cnt] * 1000
                # if u_ts_in_session_long[cnt] > 0:
                #     u_cnt_in_session_long_density[cnt] = u_cnt_in_session_long[cnt] / u_ts_in_session_long[cnt] * 1000
                
                # if u_cnt_in_session[cnt] > 0:
                #     u_corr_rate_in_session[cnt] = self.user_corr_cnt_in_session_dict[uid] / u_cnt_in_session[cnt]
                # else:
                #     u_corr_rate_in_session[cnt] = np.nan
                
                if u_cnt_in_session_short[cnt] > 0:
                    u_corr_rate_in_session_short[cnt] = self.user_corr_cnt_in_session_short_dict[uid] / u_cnt_in_session_short[cnt]
                else:
                    u_corr_rate_in_session_short[cnt] = np.nan
                
                # if u_cnt_in_session_long[cnt] > 0:
                #     u_corr_rate_in_session_long[cnt] = self.user_corr_cnt_in_session_long_dict[uid] / u_cnt_in_session_long[cnt]
                # else:
                #     u_corr_rate_in_session_long[cnt] = np.nan
                    
                # update dict
                if update_dict is True:
                    self.update_user_log_dict(
                        uid, cid, part, tags, ans, timestamp, et, hexp,
                        u_prev_difftime[cnt], u_prev_lagtime[cnt]
                    )

                cnt += 1

            user_feats_df = pd.DataFrame({
                'u_cnt': u_cnt,
                'u_corr_cnt': u_corr_cnt,
                'u_hade_cnt': u_hade_cnt,
                'u_hade_corr_cnt': u_hade_corr_cnt,
                # 'u_session_cnt': u_session_cnt,
                # 'u_session_change': u_session_change,
                'u_session_short_cnt': u_session_short_cnt,
                # 'u_session_short_change': u_session_short_change,
                # 'u_session_long_cnt': u_session_long_cnt,
                # 'u_session_long_change': u_session_long_change,
                # 'u_cnt_in_session': u_cnt_in_session,
                'u_cnt_in_session_short': u_cnt_in_session_short,
                # 'u_cnt_in_session_long': u_cnt_in_session_long,
                # 'u_cnt_density': u_cnt_density,
                # 'u_corr_cnt_density': u_corr_cnt_density,
                # 'u_session_cnt_density': u_session_cnt_density,
                # 'u_session_short_cnt_density': u_session_short_cnt_density,
                # 'u_session_long_cnt_density': u_session_long_cnt_density,
                # 'u_cnt_in_session_density': u_cnt_in_session_density,
                'u_cnt_in_session_short_density': u_cnt_in_session_short_density,
                # 'u_cnt_in_session_long_density': u_cnt_in_session_long_density,
                'u_latest1_corr_rate': u_latest1_corr_rate,
                'u_latest3_corr_rate': u_latest3_corr_rate,
                'u_latest5_corr_rate': u_latest5_corr_rate,
                'u_latest10_corr_rate': u_latest10_corr_rate,
                'u_latest30_corr_rate': u_latest30_corr_rate,
                'u_latest100_corr_rate': u_latest100_corr_rate,
                # 'u_corr_rate_in_session': u_corr_rate_in_session,
                'u_corr_rate_in_session_short': u_corr_rate_in_session_short,
                # 'u_corr_rate_in_session_long': u_corr_rate_in_session_long,
                #
                # Question #
                #
                'u_hist_ques_ansed': u_hist_ques_ansed,
                # 'u_hist_ques_corr_ansed': u_hist_ques_corr_ansed,
                'u_hist_ques_num': u_hist_ques_num,
                # 'u_hist_ques_corr_num': u_hist_ques_corr_num,
                # 'u_hist_ques_nuni':u_hist_ques_nuni,
                # 'u_hist_ques_corr_nuni':u_hist_ques_corr_nuni,
                #
                # Part #
                #
                'u_hist_part_ansed': u_hist_part_ansed,
                'u_hist_part_corr_ansed': u_hist_part_corr_ansed,
                'u_hist_part_num': u_hist_part_num,
                'u_hist_part_corr_num': u_hist_part_corr_num,
                # 'u_hist_part_nuni':u_hist_part_nuni,
                # 'u_hist_part_corr_nuni':u_hist_part_corr_nuni,
                # 'u_hist_part_std': u_hist_part_std,
                # 'u_hist_part_corr_std':u_hist_part_corr_std,
                # 'u_hist_part_rank':u_hist_part_rank,
                # どれくらい得意？ diff_from_min(mean?)
                'u_part_1_cnt': u_part_1_cnt,
                'u_part_2_cnt': u_part_2_cnt,
                'u_part_3_cnt': u_part_3_cnt,
                'u_part_4_cnt': u_part_4_cnt,
                'u_part_5_cnt': u_part_5_cnt,
                'u_part_6_cnt': u_part_6_cnt,
                'u_part_7_cnt': u_part_7_cnt,
                'u_part_1_corr_rate': u_part_1_corr_rate,
                'u_part_2_corr_rate': u_part_2_corr_rate,
                'u_part_3_corr_rate': u_part_3_corr_rate,
                'u_part_4_corr_rate': u_part_4_corr_rate,
                'u_part_5_corr_rate': u_part_5_corr_rate,
                'u_part_6_corr_rate': u_part_6_corr_rate,
                'u_part_7_corr_rate': u_part_7_corr_rate,
                #
                # Tag #
                #
                'u_hist_tag_num': u_hist_tag_num,
                'u_hist_tag_corr_num': u_hist_tag_corr_num,
                # 'u_hist_tag_nuni':u_hist_tag_nuni,
                # 'u_hist_tag_corr_nuni':u_hist_tag_corr_nuni,
                # 'u_hist_tag_max': u_hist_tag_max,
                # 'u_hist_tag_min': u_hist_tag_min,
                # 'u_hist_tag_sum': u_hist_tag_sum,
                # 'u_hist_tag_std':u_hist_tag_std,
                # 'u_hist_tag_corr_max': u_hist_tag_corr_max,
                # 'u_hist_tag_corr_min': u_hist_tag_corr_min,
                # 'u_hist_tag_corr_sum': u_hist_tag_corr_sum,
                # 'u_hist_tag_corr_std':u_hist_tag_corr_std,
                'u_hist_tag_rank_max': u_hist_tag_rank_max,
                'u_hist_tag_rank_min': u_hist_tag_rank_min,
                'u_hist_tag_rank_sum': u_hist_tag_rank_sum,
                'u_hist_tag_corr_rate_max': u_hist_tag_corr_rate_max,
                'u_hist_tag_corr_rate_min': u_hist_tag_corr_rate_min,
                'u_hist_tag_corr_rate_mn': u_hist_tag_corr_rate_mn,
                #
                # Lecture
                #
                'u_hist_lec_cnt': u_hist_lec_cnt,
                #
                # Time
                #
                'u_prev_difftime': u_prev_difftime,
                'u_prev_difftime_2': u_prev_difftime_2,
                'u_prev_difftime_3': u_prev_difftime_3,
                'u_prev_difftime_4': u_prev_difftime_4,
                'u_prev_difftime_5': u_prev_difftime_5,
                'u_prev_difftime_incorr': u_prev_difftime_incorr,
                'u_prev_lagtime': u_prev_lagtime,
                # 'u_hist_et_sum': u_hist_et_sum,
                'u_hist_et_sum_div_cnt': u_hist_et_sum_div_cnt,
                'u_hist_et_sum_div_corr_cnt': u_hist_et_sum_div_corr_cnt,
                # 'u_hist_difftime_sum': u_hist_difftime_sum,
                # 'u_hist_lagtime_sum': u_hist_lagtime_sum,
                'u_latest5_lagtime_max': u_latest5_lagtime_max,
                'u_latest5_difftime_max': u_latest5_difftime_max,
                'u_latest5_lagtime_mn': u_latest5_lagtime_mn,
                'u_latest5_difftime_mn': u_latest5_difftime_mn,
                # 'u_latest5_et_max': u_latest5_et_max,
                'u_latest5_et_mn': u_latest5_et_mn,
                'u_latest10_lagtime_max': u_latest10_lagtime_max,
                'u_latest10_difftime_max': u_latest10_difftime_max,
                'u_latest10_lagtime_mn': u_latest10_lagtime_mn,
                'u_latest10_difftime_mn': u_latest10_difftime_mn,
                'u_latest10_et_mn': u_latest10_et_mn,
                'u_latest10_et_max': u_latest10_et_max,
                # 'u_latest10_lagtime_corr_max': u_latest10_lagtime_corr_max,
                # 'u_latest10_lagtime_corr_mn': u_latest10_lagtime_corr_mn,
                # 'u_latest10_difftime_corr_max': u_latest10_difftime_corr_max,
                # 'u_latest10_difftime_corr_mn': u_latest10_difftime_corr_mn,
                # 'u_latest10_et_corr_max': u_latest10_et_corr_max,
                # 'u_latest10_et_corr_mn': u_latest10_et_corr_mn,
                # 'u_prev_difftime_norm': u_prev_difftime_norm,
                # 'u_prev_lagtime_norm': u_prev_lagtime_norm,
                # 'u_latest5_difftime_mn_norm': u_latest5_difftime_mn_norm,
                'u_prev_et_diff_time': u_prev_et_diff_time,
                'u_prev_et_lag_time': u_prev_et_lag_time,
                # 'u_ts_in_session': u_ts_in_session,
                'u_ts_in_session_short': u_ts_in_session_short,
                # 'u_ts_in_session_long': u_ts_in_session_long,
                'timestamp_lag_div_rolling5_median_each_user': timestamp_lag_div_rolling5_median_each_user,
                'timestamp_lag_div_rolling7_median_each_user': timestamp_lag_div_rolling7_median_each_user,
                'timestamp_lag_div_rolling10_median_each_user': timestamp_lag_div_rolling10_median_each_user,
                'timestamp_lag_diff_rolling5_median_each_user': timestamp_lag_diff_rolling5_median_each_user,
                'timestamp_lag_diff_rolling7_median_each_user': timestamp_lag_diff_rolling7_median_each_user,
                'timestamp_lag_diff_rolling10_median_each_user': timestamp_lag_diff_rolling10_median_each_user,
            })
            if val is True:
                user_feats_df['prev_part_s1xxxpart__answered_correctly_sm10'] = u_prev_part_sequence_te_s1
                user_feats_df['prev_part_s2xxxprev_part_s1xxxpart__answered_correctly_sm10'] = u_prev_part_sequence_te_s2
                user_feats_df['prev_part_s3xxxprev_part_s2xxxprev_part_s1xxxpart__answered_correctly_sm10'] = u_prev_part_sequence_te_s3
                # user_feats_df['prev_question_id_s1xxxquestion_id__answered_correctly_sm10'] = u_prev_ques_sequence_te_s1
                # user_feats_df['prev_question_id_s2xxxprev_question_id_s1xxxquestion_id__answered_correctly_sm10'] = u_prev_ques_sequence_te_s2
            user_feats_df = self.dataframe_process(user_feats_df)
            return user_feats_df

    def reduce_svd(self, features, n_components):
        dr_model = TruncatedSVD(n_components=n_components, random_state=46)
        features_dr = dr_model.fit_transform(features)
        return features_dr

    def fe_agg(self, X_tra_wo_lec, cols, target, table_name=''):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + [target]].groupby(cols).agg(['count', 'std']).reset_index()
        agg.columns = cols + [f'{colname}__count{table_name}', f'{colname}__std{table_name}']

#         agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__{target}_agg_count_std{table_name}.feather')
        return agg

    def fe_unique_user(self, X_tra_wo_lec, cols, table_name=''):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + ['user_id']].groupby(cols)['user_id'].nunique().reset_index()
        agg.columns = cols + [f'{colname}__unique_user{table_name}']

#         agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__unique_user{table_name}.feather')
        return agg

    def fe_te_sm(self, df, cols, target, mn, table_name=''):

        colname = 'xxx'.join(cols)
        fname = f'{colname}__{target}_sm{self.smooth}{table_name}'

        agg = df[cols + [target]].groupby(cols).agg(['mean', 'count']).reset_index()
        agg.columns = cols + [f'{colname}__{target}', f'{colname}__count']
        agg[fname] = (agg[f'{colname}__count'] * agg[f'{colname}__{target}'] + self.smooth * mn) / (agg[f'{colname}__count'] + self.smooth)
        agg = agg[cols + [fname]]

#         agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__{target}_sm{self.smooth}{table_name}.feather')
        return agg

    def fe_ent(self, X_tra_wo_lec, cols, target):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + [target]].groupby(cols)[target].agg(
            lambda x: multinomial.entropy(1, x.value_counts(normalize=True))
        ).reset_index()
        agg.columns = cols + [f'{colname}__entropy__{target}']

#         agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__entropy__{target}.feather')
        return agg

    def fe_svd(self, X_tra_wo_lec, cols):

        colname = 'xxx'.join(cols)
        svd_dim = 5
        ids = []
        sequences = []
        for c, row in tqdm(X_tra_wo_lec[['user_id'] + cols].groupby(cols)):
            ids.append(c)
            sequences.append(row['user_id'].values.tolist())
        mlb = MultiLabelBinarizer()
        tags_mlb = mlb.fit_transform(sequences)
        svd = self.reduce_svd(tags_mlb, n_components=svd_dim)
        svd = pd.DataFrame(svd).add_prefix(f'{colname}__svd_')
        if len(cols) == 1:
            svd[cols[0]] = ids
        else:
            svd[cols] = ids

#         svd.to_feather(f'../save/features_{FOLD_NAME}/{colname}__svd_feat.feather')
        return svd

    def fe_tag_te_agg(self, X_tra_wo_lec, question):

        tmp = {i: x.split() for i, x in enumerate(question['tags'].fillna('999').values)}
        tmp = {'categories': tmp}
        data2 = pd.DataFrame.from_dict(tmp)
        data3 = data2['categories'].apply(Counter)

        tag_df = pd.DataFrame.from_records(data3).fillna(value=0).astype('int8').add_prefix('tag_').reset_index().rename(columns={'index': 'question_id'})
        tag_list = tag_df.columns.values[1:].tolist()
        rows = []
        for tid in tqdm(tag_list):
            tmp = pd.merge(X_tra_wo_lec[['content_id', 'answered_correctly']], tag_df[['question_id', tid]], left_on='content_id', right_on='question_id', how='left')
            mn, cnt = tmp[tmp[tid] == 1].answered_correctly.mean(), tmp[tmp[tid] == 1].answered_correctly.count()
            rows.append([tid, mn, cnt])
        tag_stats_df = pd.DataFrame(rows, columns=['tid', 'mn', 'cnt'])
        tag_mn = {i.split('_')[-1]: j for i, j in tag_stats_df[['tid', 'mn']].values}
        tag_scores = [np.array([tag_mn[j] for j in i.split()]) for i in question['tags'].fillna('999').values]

        question['tags_max'] = [i.max() for i in tag_scores]
        question['tags_min'] = [i.min() for i in tag_scores]
        question['tags_cnt'] = [len(i) for i in tag_scores]
        question['tags_mean'] = [i.mean() for i in tag_scores]
        question['tags_std'] = [i.std() for i in tag_scores]

        content_features = []
        content_features += ['tags_max', 'tags_min', 'tags_cnt', 'tags_mean', 'tags_std']
        question = question[['question_id'] + content_features]
        question.columns = ['content_id'] + content_features

#         question.to_feather(f'../save/features_{FOLD_NAME}/content_id__tag_feat.feather')
        return question

    def fe_content_session_border_feature(self, X_tra_wo_lec):

        rows = []
        for uid, user_df in tqdm(X_tra_wo_lec[['user_id', 'content_id', 'timestamp']].groupby('user_id')):
            user_df = user_df.reset_index(drop=True)
            user_df['user_timestamp_diff'] = user_df['timestamp'].diff()
            user_df['user_timestamp_diff'] = user_df['user_timestamp_diff'].fillna('0.0').astype('float32')
            user_df['user_session_start'] = (user_df['user_timestamp_diff'] > self.session_short_th) * 1
            user_df['user_session_end'] = user_df['user_session_start'].shift(-1).fillna(0.0)
            user_df.loc[0, 'user_session_start'] = 1.0
            rows.extend(user_df.values)
        session_df = pd.DataFrame(rows, columns=['user_id', 'content_id', 'timestamp', 'user_timestamp_diff', 'user_session_start', 'user_session_end'])
        session_df = session_df.astype({
            'user_id': 'int',
            'content_id': 'int',
            'timestamp': 'int',
        })

        smooth = 10
        tar = 'user_session_start'
        mn = session_df[tar].mean()
        agg = session_df.groupby('content_id')[tar].agg(['count', 'mean']).reset_index()
        agg['mean_smooth'] = (agg['count'] * agg['mean'] + smooth * mn) / (agg['count'] + smooth)

        feat = pd.concat([
            agg[['content_id']], agg[['mean_smooth']].add_prefix(f'{tar}__')
        ], axis=1)

        tar = 'user_session_end'
        mn = session_df[tar].mean()
        agg = session_df.groupby('content_id')[tar].agg(['count', 'mean']).reset_index()
        agg['mean_smooth'] = (agg['count'] * agg['mean'] + smooth * mn) / (agg['count'] + smooth)

        feat = pd.concat([
            feat, agg[['mean_smooth']].add_prefix(f'{tar}__')
        ], axis=1)

#         feat.to_feather(f'../save/features_{FOLD_NAME}/content_session_border_feat.feather')
        return feat

    def fe_content_order_feature(self, X_tra_wo_lec):
        rows = []
        for uid, user_df in tqdm(X_tra_wo_lec[['user_id', 'content_id']].groupby('user_id')):
            user_df['order'] = user_df.reset_index(drop=True).index + 1
            rows.extend(user_df.values)
        order_df = pd.DataFrame(rows, columns=['user_id', 'content_id', 'order'])

        agg_funcs = ['mean', 'median', 'max', 'min', 'std']
        agg = order_df.groupby('content_id')['order'].agg(agg_funcs).reset_index()
        feat = pd.concat([
            agg[['content_id']], agg[agg_funcs].add_prefix('content_order_')
        ], axis=1)

#         feat.to_feather(f'../save/features_{FOLD_NAME}/content_order_feat.feather')
        return feat

    def extract_content_id_feat(
        self, X_tra_wo_lec, repeat, et_table, question,
        w2v_features, g_features, corr_g_features, ge_dw_features
    ):

        col = 'content_id'
        tar = 'answered_correctly'
        wtar = 'weighted_answered_correctly'
        ws = 'weighted_score'
        ua = 'user_answer'
        qet = 'question_elapsed_time'

        features = pd.DataFrame(X_tra_wo_lec[col].unique(), columns=[col]).sort_values(col).reset_index(drop=True)

        if f'{col}__count' in self.use_features or f'{col}__std' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__count_repeat' in self.use_features or f'{col}__std_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(repeat, [col], target=tar, table_name=table_name)
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__unique_user' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__unique_user.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_unique_user(X_tra_wo_lec, [col])
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=wtar, mn=self.train_wans_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=ws, mn=self.train_ws_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=tar, mn=self.train_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=wtar, mn=self.train_wans_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=ws, mn=self.train_ws_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, [col], target=ua)
            features = pd.merge(features, ent_feat, on=col, how='left')

        if f'{col}__entropy__answered_correctly' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{tar}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, ent_feat, on=col, how='left')

        if f'{col}__question_elapsed_time_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{qet}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(et_table, [col], target=qet, mn=et_table[qet].mean())
            features = pd.merge(features, te_feat, on=col, how='left')

        if 'tags_max' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__tag_feat.feather'
            if os.path.exists(fpath):
                tag_feat = pd.read_feather(fpath)
            else:
                tag_feat = self.fe_tag_te_agg(X_tra_wo_lec, question)
            features = pd.merge(features, tag_feat, on=col, how='left')

        if f'{col}__svd_0' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, [col])
            features = pd.merge(features, svd_feat, on=col, how='left')

        if 'w2v_0' in self.use_features:
            features = pd.merge(features, w2v_features, on=col, how='left')

        if 'gsvd_0' in self.use_features:
            features = pd.merge(features, g_features, on=col, how='left')

        if 'corr_gsvd_0' in self.use_features:
            features = pd.merge(features, corr_g_features, on=col, how='left')

        if 'ge_dw_svd_0' in self.use_features:
            features = pd.merge(features, ge_dw_features, on=col, how='left')

        if 'ge_s2v_svd_0' in self.use_features:
            features = pd.merge(features, ge_s2v_features, on=col, how='left')
        
        if 'user_session_start__mean_smooth' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/content_session_border_feat.feather'
            if os.path.exists(fpath):
                feat = pd.read_feather(fpath)
            else:
                feat = self.fe_content_session_border_feature(X_tra_wo_lec)
            features = pd.merge(features, feat, on=col, how='left')

        if 'content_order_mean' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/content_order_feat.feather'
            if os.path.exists(fpath):
                feat = pd.read_feather(fpath)
            else:
                feat = self.fe_content_order_feature(X_tra_wo_lec)
            features = pd.merge(features, feat, on=col, how='left')

        # part features
        part_feat = self.extract_part_feat(X_tra_wo_lec, repeat, et_table)
        features['part'] = features['content_id'].map(self.question2part)
        features = pd.merge(features, part_feat, on=['part'], how='left')

        feature_list = [col] + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        cols = ['content_id'] + [i for i in features.columns.sort_values().tolist() if i != 'content_id']
        self.content_id_df = features[cols]
        self.content_id_df = self.reduce_mem_usage(self.content_id_df)
        return

    def extract_part_feat(self, X_tra_wo_lec, repeat, et_table):

        col = 'part'
        tar = 'answered_correctly'
        wtar = 'weighted_answered_correctly'
        ws = 'weighted_score'
        qet = 'question_elapsed_time'

        features = pd.DataFrame(X_tra_wo_lec[col].unique(), columns=[col]).sort_values(col).reset_index(drop=True)

        if f'{col}__count' in self.use_features or f'{col}__std' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, agg_feat, on=col, how='left')

        # te
        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=wtar, mn=self.train_wans_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=ws, mn=self.train_ws_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=tar, mn=self.train_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=wtar, mn=self.train_wans_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=ws, mn=self.train_ws_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__question_elapsed_time_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{qet}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(et_table, [col], target=qet, mn=et_table[qet].mean())
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__svd_0' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, [col])
            features = pd.merge(features, svd_feat, on=col, how='left')

        return features

    def extract_content_idxxxprior_question_had_explanation_feat(self, X_tra_wo_lec, repeat, et_table):

        cols = ['content_id', 'prior_question_had_explanation']
        col = 'xxx'.join(cols)
        tar = 'answered_correctly'
        wtar = 'weighted_answered_correctly'
        ws = 'weighted_score'
        ua = 'user_answer'
        qet = 'question_elapsed_time'

        table_name = ''
        fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
        if os.path.exists(fpath):
            features = pd.read_feather(fpath)
        else:
            features = self.fe_agg(X_tra_wo_lec, cols, target=tar)
        
        if f'{col}__unique_user' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__unique_user.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_unique_user(X_tra_wo_lec, cols)
            features = pd.merge(features, agg_feat, on=cols, how='left')

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, cols, target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__weighted_answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, cols, target=wtar, mn=self.train_wans_mn)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__weighted_score_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, cols, target=ws, mn=self.train_ws_mn)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, cols, target=tar, mn=self.train_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__weighted_answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, cols, target=wtar, mn=self.train_wans_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__weighted_score_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, cols, target=ws, mn=self.train_ws_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, cols, target=ua)
            features = pd.merge(features, ent_feat, on=cols, how='left')

        if f'{col}__entropy__answered_correctly' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{tar}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, cols, target=tar)
            features = pd.merge(features, ent_feat, on=cols, how='left')

        if f'{col}__question_elapsed_time_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{qet}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(et_table, cols, target=qet, mn=et_table[qet].mean())
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__svd_0' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, cols)
            features = pd.merge(features, svd_feat, on=cols, how='left')

        feature_list = cols + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        update_cols = cols + [i for i in features.columns.sort_values().tolist() if i not in cols]
        self.content_idxxxprior_question_had_explanation_df = features[update_cols]
        self.content_idxxxprior_question_had_explanation_df = self.reduce_mem_usage(self.content_idxxxprior_question_had_explanation_df)
        return

    def extract_content_idxxxu_cnt_cat_feat(self, X_tra_wo_lec):
        cols = ['content_id', 'u_cnt_cat']
        col = 'xxx'.join(cols)
        tar = 'answered_correctly'
        ua = 'user_answer'

        table_name = ''
        fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
        if os.path.exists(fpath):
            features = pd.read_feather(fpath)
        else:
            features = self.fe_agg(X_tra_wo_lec, cols, target=tar)

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, cols, target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, cols, target=ua)
            features = pd.merge(features, ent_feat, on=cols, how='left')

        if f'{col}__svd_0' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, cols)
            features = pd.merge(features, svd_feat, on=cols, how='left')

        feature_list = cols + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        update_cols = cols + [i for i in features.columns.sort_values().tolist() if i not in cols]
        self.content_idxxxu_cnt_cat_df = features[update_cols]
        return

    def extract_content_idxxxu_hist_ques_ansed_feat(self, X_tra_wo_lec):
        cols = ['content_id', 'u_hist_ques_ansed']
        col = 'xxx'.join(cols)
        tar = 'answered_correctly'
        ua = 'user_answer'

        table_name = ''
        fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
        if os.path.exists(fpath):
            features = pd.read_feather(fpath)
        else:
            features = self.fe_agg(X_tra_wo_lec, cols, target=tar)

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, cols, target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=cols, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, cols, target=ua)
            features = pd.merge(features, ent_feat, on=cols, how='left')

        if f'{col}__svd_0' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, cols)
            features = pd.merge(features, svd_feat, on=cols, how='left')

        feature_list = cols + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        update_cols = cols + [i for i in features.columns.sort_values().tolist() if i not in cols]
        self.content_idxxxu_hist_ques_ansed_df = features[update_cols]
        return

    def extract_u_cnt_cat_feat(self, X_tra_wo_lec):

        col = 'u_cnt_cat'
        tar = 'answered_correctly'
        ua = 'user_answer'
        features = pd.DataFrame(X_tra_wo_lec[col].unique(), columns=[col]).sort_values(col).reset_index(drop=True)

        if f'{col}__count' in self.use_features or f'{col}__std' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, [col], target=ua)
            features = pd.merge(features, ent_feat, on=col, how='left')

        feature_list = [col] + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        cols = ['u_cnt_cat'] + [i for i in features.columns.sort_values().tolist() if i != 'u_cnt_cat']
        self.u_cnt_cat_df = features[cols]
        return

    def reduce_mem_usage(self, df):
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for i, col in enumerate(df.columns):
            try:
                col_type = df[col].dtype

                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float32)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
            except ValueError:
                continue

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df
