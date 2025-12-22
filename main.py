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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
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
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--temp', default=0.03, type=float)
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    
    parser.add_argument('--feat_session_gap', default=1800, type=int)

    args = parser.parse_args()
    return args

def compute_mixed_loss(seq_q, pos_t, neg_t, pos_ids, neg_ids, item_logQ, temp=0.07):
    batch_size = seq_q.size(0)
    device = seq_q.device
    neg_large = -1e9

    pos_sim = (seq_q * pos_t).sum(dim=-1, keepdim=True)
    if item_logQ is not None:
        logQ_pos = item_logQ[pos_ids].unsqueeze(1)
        pos_logits = (pos_sim / temp) - logQ_pos
    else:
        pos_logits = pos_sim / temp

    neg_sim = torch.matmul(seq_q, neg_t.transpose(-1, -2))
    if item_logQ is not None:
        logQ_cols = item_logQ[neg_ids].unsqueeze(0)
        neg_logits = (neg_sim / temp) - logQ_cols
    else:
        neg_logits = neg_sim / temp
    
    invalid_easy = pos_ids.view(batch_size, 1).eq(neg_ids.view(1, batch_size))
    neg_logits = neg_logits.masked_fill(invalid_easy, neg_large)

    hard_sim = torch.matmul(seq_q, pos_t.transpose(-1, -2))
    if item_logQ is not None:
        logQ_hard_cols = item_logQ[pos_ids].unsqueeze(0)
        hard_logits = (hard_sim / temp) - logQ_hard_cols
    else:
        hard_logits = hard_sim / temp

    mask_self = torch.eye(batch_size, device=device).bool()
    hard_logits = hard_logits.masked_fill(mask_self, neg_large)
    
    invalid_hard = pos_ids.view(batch_size, 1).eq(pos_ids.view(1, batch_size))
    hard_logits = hard_logits.masked_fill(invalid_hard, neg_large)

    logits = torch.cat([pos_logits, neg_logits, hard_logits], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)
    return loss, float(pos_logits.mean().item()), float(hard_logits.mean().item())


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    data_path = os.environ.get('TRAIN_DATA_PATH')

    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))

    args = get_args()

    dataset = MyDataset(data_path, args)
    item_logQ = torch.from_numpy(dataset.item_logQ).to(args.device)

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

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

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
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.num_epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, num_training_steps - args.warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    print("Start training with Mixed InfoNCE (1:1 Hard/Easy) + LogQ Correction")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only: break

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

                indices = (next_token_type == 1)
                hidden = seq_embs.size(2)
                seq_flat = seq_embs.view(-1, hidden)
                pos_flat = pos_embs.view(-1, hidden)
                neg_flat = neg_embs.view(-1, hidden)
                mask_flat = indices.view(-1)

                seq_q = seq_flat[mask_flat]
                pos_t = pos_flat[mask_flat]
                neg_t = neg_flat[mask_flat]
                
                pos_ids = pos.view(-1)[mask_flat]
                neg_ids = neg.view(-1)[mask_flat]

                if seq_q.size(0) == 0: continue

                loss, pos_mean, neg_mean = compute_mixed_loss(
                    seq_q, pos_t, neg_t, pos_ids, neg_ids, item_logQ, temp=args.temp
                )

                writer.add_scalar('Model/nce_pos_logits', pos_mean, global_step)
                writer.add_scalar('Model/nce_neg_logits', neg_mean, global_step)

                for param in model.item_emb.parameters():
                    loss = loss + args.l2_emb * torch.norm(param)

            log_json = json.dumps({'global_step': global_step, 'loss': float(loss.item()), 'epoch': epoch, 'time': time.time()})
            log_file.write(log_json + '\n')
            log_file.flush()

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
            scheduler.step()

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
                    seq_q = seq_embs.view(-1, hidden)[indices.view(-1)]
                    pos_t = pos_embs.view(-1, hidden)[indices.view(-1)]
                    neg_t = neg_embs.view(-1, hidden)[indices.view(-1)]
                    pos_ids = pos.view(-1)[indices.view(-1)]
                    neg_ids = neg.view(-1)[indices.view(-1)]

                    if seq_q.size(0) == 0: continue

                    loss, _, _ = compute_mixed_loss(
                        seq_q, pos_t, neg_t, pos_ids, neg_ids, item_logQ, temp=args.temp
                    )
                    valid_loss_sum += float(loss.item())
                    valid_count += 1

        valid_loss = (valid_loss_sum / valid_count) if valid_count > 0 else 0.0
        writer.add_scalar('Loss/valid', valid_loss, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
