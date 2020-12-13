import gc

import numpy as np
from tqdm import tqdm

s = 1e-3


def add_loop_feats(df, answered_correctly_sum_user_dict, attempt_content_dict, count_user_dict,
                   timestamp_diff5_dict, seq2dec_w7_dict):

    answered_correctly_sum_user_array = np.zeros(len(df), dtype=np.int32)
    attempt_content_array = np.zeros(len(df), dtype=np.int32)
    count_user_array = np.zeros(len(df), dtype=np.int32)
    answered_correctly_rolling3_array = np.zeros(len(df), dtype=np.float32)
    timestamp_diff1_array = np.zeros(len(df), dtype=np.int32)
    timestamp_diff2_array = np.zeros(len(df), dtype=np.int32)
    timestamp_diff3_array = np.zeros(len(df), dtype=np.int32)
    timestamp_diff4_array = np.zeros(len(df), dtype=np.int32)
    timestamp_diff5_array = np.zeros(len(df), dtype=np.int32)
    # seq2dec_w3_array = np.zeros(len(df), dtype=np.float32)
    seq2dec_w5_array = np.zeros(len(df), dtype=np.float32)
    seq2dec_w7_array = np.zeros(len(df), dtype=np.float32)

    default_attempt_content_array = np.zeros(13523, dtype=np.int8)
    default_timestamp_diff5_array = np.zeros(5, dtype=np.int64)
    default_seq2dec_w7_array = np.zeros(7, dtype=bool)

    for idx, row in enumerate(tqdm(df[['user_id', 'content_id', 'timestamp', 'answered_correctly']].values)):
        user_id = row[0]
        content_id = row[1]
        timestamp = row[2]
        target = row[3]

        # insert values
        answered_correctly_sum_user_array[idx] = answered_correctly_sum_user_dict.setdefault(user_id, 0)
        attempt_content_array[idx] = attempt_content_dict.setdefault(user_id, default_attempt_content_array.copy())[content_id]
        count_user_array[idx] = count_user_dict.setdefault(user_id, 0)

        answered_correctly_rolling3_array[idx] = np.mean(seq2dec_w7_dict.setdefault(user_id, default_seq2dec_w7_array.copy())[-3:])

        timestamp_diff1_array[idx] = timestamp - timestamp_diff5_dict.setdefault(user_id, default_timestamp_diff5_array.copy())[-1]
        timestamp_diff2_array[idx] = timestamp - timestamp_diff5_dict.setdefault(user_id, default_timestamp_diff5_array.copy())[-2]
        timestamp_diff3_array[idx] = timestamp - timestamp_diff5_dict.setdefault(user_id, default_timestamp_diff5_array.copy())[-3]
        timestamp_diff4_array[idx] = timestamp - timestamp_diff5_dict.setdefault(user_id, default_timestamp_diff5_array.copy())[-4]
        timestamp_diff5_array[idx] = timestamp - timestamp_diff5_dict.setdefault(user_id, default_timestamp_diff5_array.copy())[-5]

        seq2dec_feats = seq2dec_w7_dict.setdefault(user_id, default_seq2dec_w7_array.copy())

        for i, f in enumerate(seq2dec_feats[::-1]):
            if i <= 2:
                # seq2dec_w3_array[idx] += f * 10 ** -i
                seq2dec_w5_array[idx] += f * 10 ** -i
                seq2dec_w7_array[idx] += f * 10 ** -i
            elif i <= 4:
                seq2dec_w5_array[idx] += f * 10 ** -i
                seq2dec_w7_array[idx] += f * 10 ** -i
            else:
                seq2dec_w7_array[idx] += f * 10 ** -i

        # update values
        answered_correctly_sum_user_dict[user_id] += target
        attempt_content_dict[user_id][content_id] += 1
        count_user_dict[user_id] += 1

        new_timestamp_diff5_array = timestamp_diff5_dict[user_id]
        new_timestamp_diff5_array[0] = new_timestamp_diff5_array[1]
        new_timestamp_diff5_array[1] = new_timestamp_diff5_array[2]
        new_timestamp_diff5_array[2] = new_timestamp_diff5_array[3]
        new_timestamp_diff5_array[3] = new_timestamp_diff5_array[4]
        new_timestamp_diff5_array[4] = timestamp
        timestamp_diff5_dict[user_id] = new_timestamp_diff5_array

        new_seq2dec_w7_array = seq2dec_w7_dict[user_id]
        new_seq2dec_w7_array[0] = new_seq2dec_w7_array[1]
        new_seq2dec_w7_array[1] = new_seq2dec_w7_array[2]
        new_seq2dec_w7_array[2] = new_seq2dec_w7_array[3]
        new_seq2dec_w7_array[3] = new_seq2dec_w7_array[4]
        new_seq2dec_w7_array[4] = new_seq2dec_w7_array[5]
        new_seq2dec_w7_array[5] = new_seq2dec_w7_array[6]
        new_seq2dec_w7_array[6] = (target == 1)
        seq2dec_w7_dict[user_id] = new_seq2dec_w7_array

