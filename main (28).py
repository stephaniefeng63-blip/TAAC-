# main_infonce_patch.py —— InfoNCE + action_type inspect + flexible action weighting
# 已新增：时间/点击特征参数，硬负例开关 (--hard_negatives)，可选 AMP (--amp)
import argparse
import json
import os
import time
from pathlib import Path
from collections import Counter
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Stability
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)

    # InfoNCE
    parser.add_argument('--temp', default=0.03, type=float)

    # ===== action weighting config =====
    parser.add_argument(
        '--action_weight_map',
        default='',
        type=str,
        help='optional static mapping "act:weight,act:weight", e.g. "0:1.0,1:3.1". If empty, use dynamic weighting_method.'
    )
    parser.add_argument(
        '--weighting_method',
        default='sqrt_inv_freq',
        choices=['sqrt_inv_freq', 'inv_freq', 'balanced', 'pos_only'],
        help='dynamic per-batch/global weighting strategy when action_weight_map not provided'
    )
    parser.add_argument('--beta', default=0.5, type=float, help='exponent for sqrt_inv_freq (beta=0.5 -> sqrt)')
    parser.add_argument('--clip_weight_min', default=0.1, type=float)
    parser.add_argument('--clip_weight_max', default=5.0, type=float)
    parser.add_argument('--normalize_action_weight', action='store_true', help='normalize weights to mean=1 per batch')
    parser.set_defaults(normalize_action_weight=True)

    # global EMA click rate (stability)
    parser.add_argument('--use_global_pos_ema', action='store_true', help='use EMA of global click rate to compute weights')
    parser.set_defaults(use_global_pos_ema=True)
    parser.add_argument('--ema_alpha', default=0.01, type=float, help='EMA smoothing coefficient for pos rate')

    # temperature boost for high-weight actions (optional)
    parser.add_argument('--action_temp_boost', default=0.0, type=float,
                        help='>0: lower effective temperature for higher-weight samples (sharpen positives)')

    # ===== action type inspection / logging =====
    parser.add_argument('--inspect_action_types', action='store_true', help='print & log action_type distribution by batch/epoch')
    parser.set_defaults(inspect_action_types=True)
    parser.add_argument('--print_action_types_every', default=50, type=int, help='print batch distribution every N train steps')

    # ===== 新增：时间/点击侧特征参数（dataset 使用） =====
    parser.add_argument('--feat_session_gap', default=1800, type=int, help='seconds defining a session split')
    parser.add_argument('--feat_ctr_short_k', default=5, type=int, help='recent short-window CTR length')
    parser.add_argument('--feat_ctr_long_k', default=50, type=int, help='recent long-window CTR length')

    # ===== 新增：硬负例开关 & AMP =====
    parser.add_argument('--hard_negatives', action='store_true', help='use in-batch positives as negatives (NxN InfoNCE)')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')

    args = parser.parse_args()
    return args


def parse_weight_map(map_str):
    """
    '1:1.0,2:1.2' -> {1:1.0, 2:1.2}
    """
    out = {}
    if not map_str:
        return out
    for kv in map_str.split(','):
        if not kv:
            continue
        parts = kv.split(':')
        if len(parts) != 2:
            continue
        k, v = parts
        try:
            out[int(k.strip())] = float(v.strip())
        except:
            continue
    return out


def compute_infonce_loss(seq_q, pos_t, neg_pool, temp=0.07, eps=1e-8, sample_weights=None, temp_boost=0.0):
    """
    标准版：每个样本独立的 in-batch neg_pool（通常较“易”）
    seq_q: [N, H]
    pos_t: [N, H]
    neg_pool: [N, H] or [M, H]
    sample_weights: [N] or None
    """
    # normalize vectors
    seq_q = seq_q / seq_q.norm(dim=-1, keepdim=True).clamp(min=eps)
    pos_t = pos_t / pos_t.norm(dim=-1, keepdim=True).clamp(min=eps)
    neg_pool = neg_pool / neg_pool.norm(dim=-1, keepdim=True).clamp(min=eps)

    pos_logits = F.cosine_similarity(seq_q, pos_t, dim=-1).unsqueeze(1)  # [N,1]
    neg_logits = torch.matmul(seq_q, neg_pool.transpose(-1, -2))  # [N,M]
    logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1+M]

    # temperature: support per-sample if temp_boost>0 and sample_weights provided
    if temp_boost > 0.0 and (sample_weights is not None):
        w = sample_weights.clamp(min=0.0)
        eff_temp = temp / (1.0 + temp_boost * (w - 1.0)).clamp(min=1e-6)  # [N]
        logits = logits / eff_temp.unsqueeze(1)
    else:
        logits = logits / temp

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positives are index 0

    ce_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [N]

    if sample_weights is not None:
        denom = sample_weights.sum().clamp_min(1e-6)
        loss = (ce_per_sample * sample_weights).sum() / denom
    else:
        loss = ce_per_sample.mean()

    return loss, float(pos_logits.mean().item()), float(neg_logits.mean().item())


def compute_infonce_loss_hard(seq_q, pos_t, temp=0.07, sample_weights=None):
    """
    硬负例版：使用整个 batch 的正样本作为负样本（除对角），更难
    seq_q: [N, H], pos_t: [N, H]
    """
    seq_q = F.normalize(seq_q, dim=-1)
    pos_t = F.normalize(pos_t, dim=-1)

    logits = torch.matmul(seq_q, pos_t.transpose(-1, -2)) / temp  # [N,N]
    labels = torch.arange(seq_q.size(0), device=seq_q.device)

    ce_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [N]
    if sample_weights is not None:
        loss = (ce_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
    else:
        loss = ce_per_sample.mean()

    # 便于监控的统计
    pos_mean = float((seq_q * pos_t).sum(dim=-1).mean().item())
    with torch.no_grad():
        n = logits.size(0)
        if n > 1:
            all_sim = seq_q @ pos_t.T
            diag_sum = (seq_q * pos_t).sum(dim=-1).sum()
            neg_mean = float((all_sim.sum() - diag_sum) / (n * (n - 1)))
        else:
            neg_mean = 0.0
    return loss, pos_mean, neg_mean


def compute_action_weights_from_flat(
    act_flat,
    method='sqrt_inv_freq',
    beta=0.5,
    clip_min=0.1,
    clip_max=5.0,
    global_pos_rate=None,
    normalize=True
):
    """
    act_flat: torch tensor of shape [N], values like 0/1
    method: 'sqrt_inv_freq' (default), 'inv_freq', 'balanced', 'pos_only'
    returns: weights tensor [N] (float), p_pos (float)
    """
    device = act_flat.device
    N = act_flat.numel()
    if N == 0:
        return None, 0.0

    # compute batch positive rate
    batch_pos = float((act_flat == 1).sum().item()) / max(1, N)
    p_pos = float(global_pos_rate) if (global_pos_rate is not None) else batch_pos
    p_pos = max(p_pos, 1e-6)
    p_neg = max(1.0 - p_pos, 1e-6)

    if method == 'sqrt_inv_freq':
        ratio = (p_neg / p_pos) ** beta
        w_pos = float(ratio)
        w_neg = 1.0
    elif method == 'inv_freq':
        w_pos = float(p_neg / p_pos)
        w_neg = 1.0
    elif method == 'balanced':
        w_pos = 1.0 / p_pos
        w_neg = 1.0 / p_neg
        # normalize to mean 1:
        mean_raw = w_pos * p_pos + w_neg * p_neg
        w_pos = w_pos / mean_raw
        w_neg = w_neg / mean_raw
    elif method == 'pos_only':
        if p_pos > 0:
            w_pos = 1.0 / p_pos
            w_neg = 0.0
        else:
            w_pos = 1.0
            w_neg = 1.0
    else:
        raise ValueError('unknown method')

    # build tensor weights
    w_pos_t = torch.tensor(w_pos, dtype=torch.float, device=device)
    w_neg_t = torch.tensor(w_neg, dtype=torch.float, device=device)
    weights = torch.where(act_flat == 1, w_pos_t, w_neg_t).float()

    # clip range
    weights = torch.clamp(weights, clip_min, clip_max)

    # normalize to mean=1 if requested
    if normalize:
        s = weights.sum().clamp_min(1e-6)
        weights = weights * (weights.numel() / s)

    return weights, p_pos


if __name__ == '__main__':
    # Prepare IO dirs
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # open log & writer
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))

    args = get_args()
    # parse static map (if provided)
    action_weight_map = parse_weight_map(args.action_weight_map)
    if action_weight_map:
        print(f'[ActionWeightMap] static map provided: {action_weight_map}')
    else:
        print(f'[ActionWeightMap] no static map, dynamic method: {args.weighting_method}')

    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # parameter initialization
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # zero pad embeddings
    try:
        model.pos_emb.weight.data[0, :] = 0
        model.item_emb.weight.data[0, :] = 0
        model.user_emb.weight.data[0, :] = 0
        for k in model.sparse_emb:
            model.sparse_emb[k].weight.data[0, :] = 0
    except Exception:
        pass

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 优化器 + 学习率调度器（warmup -> 余弦）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.num_epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, num_training_steps - args.warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0

    # counters for action_type inspect
    train_epoch_counter = Counter()
    valid_epoch_counter = Counter()
    # global EMA for pos rate (if enabled)
    pos_rate_ema = None

    print("Start training (InfoNCE; action_type inspection={})".format(args.inspect_action_types))

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        # reset per-epoch statistics
        train_epoch_counter.clear()
        valid_epoch_counter.clear()

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                seq_embs, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )

                # mask: positions where next_token_type == 1
                indices = (next_token_type == 1)
                batch_size = seq_embs.size(0)
                seq_len = seq_embs.size(1)
                hidden = seq_embs.size(2)

                seq_flat = seq_embs.view(-1, hidden)
                pos_flat = pos_embs.view(-1, hidden)
                neg_flat = neg_embs.view(-1, hidden)
                mask_flat = indices.view(-1)

                seq_q = seq_flat[mask_flat]
                pos_t = pos_flat[mask_flat]
                neg_pool = neg_flat[mask_flat]  # in-batch negatives (较易)

                if seq_q.size(0) == 0:
                    continue

                # ========== inspect action types for this batch ==========
                act_flat = next_action_type.view(-1)[mask_flat]  # [N]
                if args.inspect_action_types:
                    # record histogram to tensorboard (as floats)
                    writer.add_histogram('ActionType/train_batch', act_flat.float(), global_step)
                    # update counters & periodic print
                    uniq, cnt = act_flat.unique(return_counts=True)
                    batch_dist = {int(uniq[i].item()): int(cnt[i].item()) for i in range(len(uniq))}
                    train_epoch_counter.update(batch_dist)
                    if (step % max(1, args.print_action_types_every)) == 0:
                        print(f'[ActionType][train][epoch={epoch} step={step}] {batch_dist}')
                        log_file.write(json.dumps({
                            'tag': 'train_batch_action_dist',
                            'epoch': epoch, 'step': step, 'dist': batch_dist, 'time': time.time()
                        }) + '\n')

                # ========== compute sample weights ==========
                sample_weights = None
                if action_weight_map:
                    weights = torch.ones_like(act_flat, dtype=torch.float, device=act_flat.device)
                    unique_vals = act_flat.unique()
                    for v in unique_vals:
                        w = action_weight_map.get(int(v.item()), 1.0)
                        weights = torch.where(act_flat == v, torch.full_like(weights, float(w)), weights)
                    if args.normalize_action_weight:
                        weights = weights * (weights.numel() / weights.sum().clamp_min(1e-6))
                    sample_weights = weights
                else:
                    global_rate = pos_rate_ema if (args.use_global_pos_ema and pos_rate_ema is not None) else None
                    weights, batch_p_pos = compute_action_weights_from_flat(
                        act_flat,
                        method=args.weighting_method,
                        beta=args.beta,
                        clip_min=args.clip_weight_min,
                        clip_max=args.clip_weight_max,
                        global_pos_rate=global_rate,
                        normalize=args.normalize_action_weight
                    )
                    sample_weights = weights
                    # update EMA
                    if args.use_global_pos_ema:
                        if pos_rate_ema is None:
                            pos_rate_ema = batch_p_pos
                        else:
                            pos_rate_ema = args.ema_alpha * batch_p_pos + (1.0 - args.ema_alpha) * pos_rate_ema

                # log weight stats
                if sample_weights is not None:
                    writer.add_scalar('ActionWeight/mean', float(sample_weights.mean().item()), global_step)
                    writer.add_scalar('ActionWeight/max', float(sample_weights.max().item()), global_step)
                    writer.add_scalar('ActionWeight/min', float(sample_weights.min().item()), global_step)
                if pos_rate_ema is not None:
                    writer.add_scalar('ActionWeight/pos_rate_ema', float(pos_rate_ema), global_step)

                # ========== compute InfoNCE loss (weighted) ==========
                if args.hard_negatives:
                    loss, pos_mean, neg_mean = compute_infonce_loss_hard(
                        seq_q, pos_t, temp=args.temp, sample_weights=sample_weights
                    )
                else:
                    loss, pos_mean, neg_mean = compute_infonce_loss(
                        seq_q, pos_t, neg_pool, temp=args.temp,
                        sample_weights=sample_weights, temp_boost=args.action_temp_boost
                    )

                writer.add_scalar('Model/nce_pos_logits', pos_mean, global_step)
                writer.add_scalar('Model/nce_neg_logits', neg_mean, global_step)

                # embedding L2 regularization
                for param in model.item_emb.parameters():
                    loss = loss + args.l2_emb * torch.norm(param)

            log_json = json.dumps({'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch, 'time': time.time()})
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', float(loss.item()), global_step)

            global_step += 1

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                optimizer.step()
            scheduler.step()  # warmup/decay step

        # end train epoch: save action_type train epoch summary
        if args.inspect_action_types and len(train_epoch_counter) > 0:
            total = sum(train_epoch_counter.values())
            summary = {k: {'count': int(v), 'ratio': float(v) / total} for k, v in sorted(train_epoch_counter.items())}
            print(f'[ActionType][train][epoch={epoch}] total={total} summary={summary}')
            with open(Path(os.environ.get('TRAIN_LOG_PATH'), f'action_type_train_epoch{epoch}.json'), 'w') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        # ========== validation ==========
        model.eval()
        valid_loss_sum = 0.0
        valid_count = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                token_type = token_type.to(args.device)
                next_token_type = next_token_type.to(args.device)
                next_action_type = next_action_type.to(args.device)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    seq_embs, pos_embs, neg_embs = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )

                    indices = (next_token_type == 1)
                    hidden = seq_embs.size(2)
                    seq_flat = seq_embs.view(-1, hidden)
                    pos_flat = pos_embs.view(-1, hidden)
                    neg_flat = neg_embs.view(-1, hidden)
                    mask_flat = indices.view(-1)

                    seq_q = seq_flat[mask_flat]
                    pos_t = pos_flat[mask_flat]
                    neg_pool = neg_flat[mask_flat]

                    if seq_q.size(0) == 0:
                        continue

                    # inspect valid batch action distribution
                    act_flat = next_action_type.view(-1)[mask_flat]
                    if args.inspect_action_types:
                        writer.add_histogram('ActionType/valid_batch', act_flat.float(), global_step)
                        uniq, cnt = act_flat.unique(return_counts=True)
                        batch_dist = {int(uniq[i].item()): int(cnt[i].item()) for i in range(len(uniq))}
                        valid_epoch_counter.update(batch_dist)

                    # compute sample_weights same as training
                    sample_weights = None
                    if action_weight_map:
                        weights = torch.ones_like(act_flat, dtype=torch.float, device=act_flat.device)
                        unique_vals = act_flat.unique()
                        for v in unique_vals:
                            w = action_weight_map.get(int(v.item()), 1.0)
                            weights = torch.where(act_flat == v, torch.full_like(weights, float(w)), weights)
                        if args.normalize_action_weight:
                            weights = weights * (weights.numel() / weights.sum().clamp_min(1e-6))
                        sample_weights = weights
                    else:
                        global_rate = pos_rate_ema if (args.use_global_pos_ema and pos_rate_ema is not None) else None
                        weights, batch_p_pos = compute_action_weights_from_flat(
                            act_flat,
                            method=args.weighting_method,
                            beta=args.beta,
                            clip_min=args.clip_weight_min,
                            clip_max=args.clip_weight_max,
                            global_pos_rate=global_rate,
                            normalize=args.normalize_action_weight
                        )
                        sample_weights = weights

                    if args.hard_negatives:
                        loss, pos_mean, neg_mean = compute_infonce_loss_hard(
                            seq_q, pos_t, temp=args.temp, sample_weights=sample_weights
                        )
                    else:
                        loss, pos_mean, neg_mean = compute_infonce_loss(
                            seq_q, pos_t, neg_pool, temp=args.temp,
                            sample_weights=sample_weights, temp_boost=args.action_temp_boost
                        )

                    valid_loss_sum += float(loss.item())
                    valid_count += 1

        valid_loss = (valid_loss_sum / valid_count) if valid_count > 0 else 0.0
        writer.add_scalar('Loss/valid', valid_loss, global_step)

        # save action_type valid epoch summary
        if args.inspect_action_types and len(valid_epoch_counter) > 0:
            total = sum(valid_epoch_counter.values())
            summary = {k: {'count': int(v), 'ratio': float(v) / total} for k, v in sorted(valid_epoch_counter.items())}
            print(f'[ActionType][valid][epoch={epoch}] total={total} summary={summary}')
            with open(Path(os.environ.get('TRAIN_LOG_PATH'), f'action_type_valid_epoch{epoch}.json'), 'w') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        # checkpoint
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
