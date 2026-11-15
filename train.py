# -*- coding: utf-8 -*-
"""实验四 训练脚本：BCE + Dice 联合损失，按 FOV 内 Dice 选模型"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VesselDataset, collate_path
from model import UNetVessel


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def build_loss(logits, target, cfg):
    """BCE 解决像素级分类稳定性，Dice 直接优化重叠度，两者互补"""
    bce = F.binary_cross_entropy_with_logits(
        logits, target,
        pos_weight=torch.tensor([cfg['pos_weight']], device=logits.device))
    d = dice_loss(logits, target)
    return cfg['w_bce'] * bce + cfg['w_dice'] * d, bce.item(), d.item()


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    tp = fp = fn = tn = 0.0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch['image'])
        prob = torch.sigmoid(logits)
        pred = (prob > cfg['threshold']).float() * batch['fov']
        gt = batch['mask'] * batch['fov']
        tp += float((pred * gt).sum())
        fp += float((pred * (1 - gt)).sum())
        fn += float(((1 - pred) * gt).sum())
        tn += float(((1 - pred) * (1 - gt)).sum())
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {'dice': dice, 'sensitivity': sens, 'specificity': spec, 'accuracy': acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/train_config.json')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--no_attention', action='store_true', help='消融：关闭注意力门控')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = VesselDataset(args.train_csv, args.size, train=True)
    val_ds = VesselDataset(args.val_csv, args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_path)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate_path)

    model = UNetVessel(use_attention=not args.no_attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                          eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best = {'dice': -1.0}
    bad = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc='ep%03d' % ep, ncols=110)
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(batch['image'])
                loss, bce_v, dice_v = build_loss(logits.float(), batch['mask'], cfg)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * batch['image'].size(0)
            n += batch['image'].size(0)
            pbar.set_postfix(loss='%.4f' % loss.item())
        scheduler.step()

        m = evaluate(model, val_loader, device, cfg)
        print('[ep%03d] train_loss=%.4f dice=%.4f sens=%.4f spec=%.4f acc=%.4f'
              % (ep, running / n, m['dice'], m['sensitivity'], m['specificity'],
                 m['accuracy']))

        if m['dice'] > best['dice']:
            best = dict(m)
            best['epoch'] = ep
            bad = 0
            torch.save({'model': model.state_dict(), 'size': args.size,
                        'cfg': cfg, 'metrics': dict(m)}, 'best_vessel_model.pth')
            print('  -> 保存 best_vessel_model.pth (dice=%.4f)' % m['dice'])
        else:
            bad += 1
            if bad >= args.patience:
                print('验证 Dice %d 轮未提升，提前停止' % args.patience)
                break

    print('训练结束，最优 Dice %.4f（敏感度 %.4f，特异度 %.4f，epoch %d）'
          % (best['dice'], best['sensitivity'], best['specificity'], best['epoch']))


if __name__ == '__main__':
    main()
