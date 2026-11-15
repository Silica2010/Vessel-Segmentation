# -*- coding: utf-8 -*-
"""实验四 推理：滑窗分割 + 按任务要求输出（血管 0，其他 255）

为什么不直接整图缩放后推理：血管宽度往往只有 1~2 像素，缩到 512 会把细血管抹掉，
因此这里在原分辨率上用重叠滑窗推理，再把概率图拼回原尺寸。
"""

import argparse

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD, preprocess
from model import UNetVessel


def sliding_window_predict(model, img, patch=512, stride=384, device='cpu'):
    """返回原分辨率下的血管概率图"""
    h, w = img.shape[:2]
    prob = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - patch + 1), stride)) + [max(0, h - patch)]
    xs = list(range(0, max(1, w - patch + 1), stride)) + [max(0, w - patch)]
    ys, xs = sorted(set(ys)), sorted(set(xs))

    for y in ys:
        for x in xs:
            crop = img[y:y + patch, x:x + patch]
            if crop.shape[0] != patch or crop.shape[1] != patch:
                crop = cv2.resize(crop, (patch, patch))
            rgb = crop.astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            prob[y:y + patch, x:x + patch] += p
            count[y:y + patch, x:x + patch] += 1.0
    return prob / np.maximum(count, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--weights', type=str, default='best_vessel_model.pth')
    ap.add_argument('--output', type=str, default='vessel_mask.png')
    ap.add_argument('--patch', type=int, default=512)
    ap.add_argument('--stride', type=int, default=384)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)
    model = UNetVessel().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    raw = cv2.imread(args.image)
    if raw is None:
        raise FileNotFoundError('无法读取图像: %s' % args.image)
    img = preprocess(raw)
    prob = sliding_window_predict(model, img, args.patch, args.stride, device)

    binary = (prob > args.threshold).astype(np.uint8)
    vessel_ratio = float(binary.sum()) / binary.size
    print('血管像素占比：%.4f（参考范围约 0.05~0.15）' % vessel_ratio)

    # 任务要求：血管 0，其他 255
    out = np.where(binary > 0, 0, 255).astype(np.uint8)
    cv2.imwrite(args.output, out)
    print('已保存：%s，尺寸 %dx%d' % (args.output, out.shape[1], out.shape[0]))
    print('取值检查：唯一值 %s' % np.unique(out).tolist())


if __name__ == '__main__':
    main()
