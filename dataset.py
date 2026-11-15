# -*- coding: utf-8 -*-
"""实验四 数据集：眼底图 + 血管掩膜

csv 约定：image_path,mask_path
- 掩膜允许两种标注惯例：血管为 255 或为 1，统一按 > 127 阈值二值化；
- 内部标签约定 vessel = 1；导出时再按任务要求反相为“血管 0，其他 255”。
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fundus_fov(img_bgr):
    """从眼底图推算视野（FOV）掩膜：亮度阈值 + 最大连通域，用于排除黑边。

    黑边如果参与评估，会被误判成“大片正确背景”，指标虚高。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(fov, [largest], -1, 255, thickness=-1)
    return fov


def preprocess(img_bgr):
    """绿通道对比度最好（血管吸收红光），CLAHE 强化细血管"""
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = clahe.apply(green)
    rgb = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    return rgb


class VesselDataset(Dataset):
    def __init__(self, csv_path, size=512, train=False):
        self.size = size
        self.train = train
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            if header[:2] != ['image_path', 'mask_path']:
                raise ValueError('csv 列名应为 image_path,mask_path，实际为 %s' % header)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split(',')[:2]
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def _augment(self, img, mask):
        """只做几何增强：血管是低对比度结构，颜色扰动会把它和背景混在一起"""
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError('无法读取: %s / %s' % (img_path, mask_path))

        raw = img.copy()
        img = preprocess(img)
        fov = fundus_fov(raw)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        fov = cv2.resize(fov, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, mask = self._augment(img, mask)

        mask = (mask > 127).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return {
            'image': torch.from_numpy(img.transpose(2, 0, 1)),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'fov': torch.from_numpy((fov > 0).astype(np.float32)).unsqueeze(0),
            'path': img_path,
        }


def collate_path(batch):
    out = {}
    for key in batch[0].keys():
        if key == 'path':
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
