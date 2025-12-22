from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset import save_emb

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None
            )
        else:
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)
        return output, None

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        return outputs

class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        self.time_dense_keys = [
            'u_sin_hour','u_cos_hour','u_sin_dow','u_cos_dow',
            'u_log_time_gap','u_session_pos','u_smooth_ctr'
        ]
        time_dense_dim = len(self.time_dense_keys)

        bucket_dim = max(1, args.hidden_units // 8)
        self.time_bucket_info = {
            'u_hour_i': (24,  bucket_dim),
            'u_dow_i':  (7,   bucket_dim),
            'u_month_i':(12,  bucket_dim),
            'u_week_i': (54,  bucket_dim),
            'u_year_i': (41,  bucket_dim),
            'u_time_bucket6_i': (6, bucket_dim),
            'u_is_weekend_i':   (2, bucket_dim),
        }
        self.time_bucket_embs = torch.nn.ModuleDict({
            k: torch.nn.Embedding(vocab + 1, dim, padding_idx=0)
            for k, (vocab, dim) in self.time_bucket_info.items()
        })
        time_feature_dim = time_dense_dim + sum(dim for (_, dim) in self.time_bucket_info.values())

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(self.USER_CONTINUAL_FEAT)
        
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        itemdim_with_time = itemdim + time_feature_dim

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim_with_time, args.hidden_units) 

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        for k in self.ITEM_SPARSE_FEAT:
            tensor_feature = self.feat2tensor(feature_array, k)
            item_feat_list.append(self.sparse_emb[k](tensor_feature))
        for k in self.ITEM_ARRAY_FEAT:
            tensor_feature = self.feat2tensor(feature_array, k)
            item_feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
        for k in self.ITEM_CONTINUAL_FEAT:
            tensor_feature = self.feat2tensor(feature_array, k)
            item_feat_list.append(tensor_feature.unsqueeze(2).float())
        
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0]) if batch_size > 0 else 0
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for bi in range(batch_size):
                seq_feats = feature_array[bi]
                for tj in range(seq_len):
                    if k in seq_feats[tj]:
                        batch_emb_data[bi, tj] = seq_feats[tj][k]
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        batch_size = len(feature_array)
        seq_len = len(feature_array[0]) if batch_size > 0 else 0
        
        dense_arr = np.zeros((batch_size, seq_len, len(self.time_dense_keys)), dtype=np.float32)
        for bi in range(batch_size):
            seq_feats = feature_array[bi]
            for tj in range(seq_len):
                feat = seq_feats[tj]
                for kk, key in enumerate(self.time_dense_keys):
                    dense_arr[bi, tj, kk] = float(feat.get(key, 0.0))
        time_dense = torch.from_numpy(dense_arr).to(self.dev)
        item_feat_list.append(time_dense)

        for k in self.time_bucket_info:
            idx_np = np.zeros((batch_size, seq_len), dtype=np.int64)
            for bi in range(batch_size):
                seq_feats = feature_array[bi]
                for tj in range(seq_len):
                    idx_np[bi, tj] = int(seq_feats[tj].get(k, 0))
            idx_tensor = torch.from_numpy(idx_np).to(self.dev)
            bucket_emb = self.time_bucket_embs[k](idx_tensor)
            item_feat_list.append(bucket_emb)
        
        if include_user:
            for k in self.USER_SPARSE_FEAT:
                tensor_feature = self.feat2tensor(feature_array, k)
                user_feat_list.append(self.sparse_emb[k](tensor_feature))
            for k in self.USER_ARRAY_FEAT:
                tensor_feature = self.feat2tensor(feature_array, k)
                user_feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
            for k in self.USER_CONTINUAL_FEAT:
                tensor_feature = self.feat2tensor(feature_array, k)
                user_feat_list.append(tensor_feature.unsqueeze(2).float())

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature):
        log_feats = self.log2feats(user_item, mask, seq_feature)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        return log_feats, pos_embs, neg_embs

    def predict(self, log_seqs, seq_feature, mask):
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        final_feat = log_feats[:, -1, :]
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])
            batch_feat = np.array(batch_feat, dtype=object)
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
