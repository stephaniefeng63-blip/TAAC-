# dataset.py （已改动：在派生 derived 时新增离散时间桶；其余流程不变）
# 变更要点：
# 1) 在 __getitem__ 里派生 derived 的同时，新增以下离散索引（1-based；0=缺失）：
#    - u_hour_i(1..24), u_dow_i(1..7), u_month_i(1..12), u_week_i(1..54), u_year_i(1..41),
#      u_time_bucket6_i(1..6，四小时分桶), u_is_weekend_i(1/2)
# 2) 这些键仅在 model.feat2emb 的时间支路里手动读取，不加入 feature_types，
#    因此不影响 indexer 与 fill_missing_feat 的默认补全逻辑。
# 3) 原有的连续时间/点击侧（8个）仍保留在 user_continual 中。
# ============================

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 可配置：会话切分阈值 & CTR 窗口
        self.session_gap = getattr(args, 'feat_session_gap', 1800)  # 30 min
        self.ctr_short_k = getattr(args, 'feat_ctr_short_k', 5)
        self.ctr_long_k = getattr(args, 'feat_ctr_long_k', 50)

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        ""
        user_sequence = self._load_user_data(uid)
        from collections import deque
        import math
        import time as _time

        derived_list = []
        dq_s = deque(maxlen=self.ctr_short_k)
        dq_l = deque(maxlen=self.ctr_long_k)

        prev_ts = None
        session_pos = 0

        def safe_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        def ctr_from_deque(d):
            return float(sum(d)) / float(len(d)) if len(d) > 0 else 0.0

        for rec in user_sequence:
            # 统一解析：长度不够就补 0
            u = rec[0] if len(rec) > 0 else 0
            i = rec[1] if len(rec) > 1 else 0
            user_feat = rec[2] if len(rec) > 2 else {}
            item_feat = rec[3] if len(rec) > 3 else {}
            act_raw = rec[4] if len(rec) > 4 else 0
            ts_raw = rec[5] if len(rec) > 5 else 0

            # action 强制二值
            act = 1 if safe_int(act_raw, 0) == 1 else 0
            # ts 容错
            ts = safe_int(ts_raw, 0)

            # 会话切分（拿不到 ts 时，所有 ts=0，视为每条都是新会话，session_pos=0）
            if (prev_ts is None) or (ts - (prev_ts or 0) > self.session_gap):
                session_pos = 0
            else:
                session_pos += 1
            prev_ts = ts if ts > 0 else prev_ts

            # 时间特征（若 ts=0，全置 0，不引噪声）
            if ts > 0:
                hour = (ts // 3600) % 24
                dow = ((ts // 86400) + 4) % 7  # 1970-01-01 是周四(+4)
                sin_h = math.sin(2 * math.pi * hour / 24.0)
                cos_h = math.cos(2 * math.pi * hour / 24.0)
                sin_w = math.sin(2 * math.pi * dow / 7.0)
                cos_w = math.cos(2 * math.pi * dow / 7.0)

                if len(derived_list) == 0:
                    log_gap = 0.0
                else:
                    prev_ts_raw = user_sequence[len(derived_list) - 1][5] if len(
                        user_sequence[len(derived_list) - 1]) > 5 else 0
                    gap = max(0, ts - safe_int(prev_ts_raw, 0))
                    # 将 gap 截断到 7 天并做 log1p，再按 7 天归一
                    log_gap = math.log1p(min(gap, 86400 * 7)) / math.log1p(86400 * 7)

                # 离散时间桶（全部 1-based；0=缺失）
                g = _time.gmtime(ts)        # 如需本地时区改 gmtime 为 localtime
                month_i = int(g.tm_mon)     # 1..12
                week_i  = int(_time.strftime("%U", g)) + 1  # 1..54（含第 0 周→1）
                year_idx = max(0, min(40, int(g.tm_year) - 2000)) + 1  # 1..41（2000..2040）
                hour_i  = int(hour) + 1     # 1..24
                dow_i   = int(dow) + 1      # 1..7
                time_bucket6_i = int(hour // 4) + 1  # 1..6（四小时分桶）
                is_weekend_i = 2 if dow in (5, 6) else 1
            else:
                sin_h = cos_h = sin_w = cos_w = 0.0
                log_gap = 0.0
                month_i = week_i = year_idx = hour_i = dow_i = time_bucket6_i = is_weekend_i = 0

            # CTR（只依赖 act）
            ctr_s = ctr_from_deque(dq_s)
            ctr_l = ctr_from_deque(dq_l)

            derived_list.append({
                'u_sin_hour': sin_h, 'u_cos_hour': cos_h,
                'u_sin_dow': sin_w,  'u_cos_dow': cos_w,
                'u_log_time_gap': float(log_gap),
                'u_recent_ctr_s': float(ctr_s),
                'u_recent_ctr_l': float(ctr_l),
                'u_session_pos': float(min(session_pos, 100) / 100.0),  # 截断后归一

                # 新增的“离散桶”（索引型，1-based；0=缺失）
                'u_hour_i': int(hour_i),
                'u_dow_i': int(dow_i),
                'u_month_i': int(month_i),
                'u_week_i': int(week_i),
                'u_year_i': int(year_idx),
                'u_time_bucket6_i': int(time_bucket6_i),
                'u_is_weekend_i': int(is_weekend_i),
            })

            dq_s.append(act)
            dq_l.append(act)

        # ---- 2) 构造扩展序列，把 derived 合并进 feat ----
        ext_user_sequence = []
        for idx, record_tuple in enumerate(user_sequence):
            u = record_tuple[0] if len(record_tuple) > 0 else 0
            i = record_tuple[1] if len(record_tuple) > 1 else 0
            user_feat = record_tuple[2] if len(record_tuple) > 2 else {}
            item_feat = record_tuple[3] if len(record_tuple) > 3 else {}
            action_type = record_tuple[4] if len(record_tuple) > 4 else 0
            ts = safe_int(record_tuple[5], 0) if len(record_tuple) > 5 else 0
            derived = derived_list[idx]

            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts, derived))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts, derived))

        # 注意：下游三元组/掩码/采样逻辑
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts_set = set()
        for rec in ext_user_sequence:
            if rec[2] == 1 and rec[0]:
                ts_set.add(rec[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, ts, derived = record_tuple
            next_i, next_feat, next_type, next_act_type, next_ts, _ = nxt

            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)

            # 注入新增连续/离散时间特征（若缺失会被默认值覆盖为 0/0.0）
            for k, v in derived.items():
                feat[k] = v

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts_set)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}

        # 稀疏/多值/向量 与原始一致
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112',  '115', '122', '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids

        # === 已有：用户侧连续特征槽位（默认 0.0） ===
        feat_types['user_continual'] = [
            'u_sin_hour', 'u_cos_hour',
            'u_sin_dow', 'u_cos_dow',
            'u_log_time_gap',
            'u_recent_ctr_s', 'u_recent_ctr_l',
            'u_session_pos'
        ]
        feat_types['item_continual'] = []

        # 默认值与统计
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])

        # 连续特征默认值
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0

        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:
                    user_id = u
                else:
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict
