# -*- coding: utf-8 -*-
"""实验四 模型：U-Net（带注意力门控）用于眼底血管二分类分割

设计要点
--------
1. 血管是细长结构，宽度常只有 1~3 像素，因此不做激进下采样：编码器最深 4 级（1/16），
   并在解码侧用跳跃连接把高分辨率边缘信息接回来。
2. 注意力门控（AG）在拼接前对跳跃特征做空间加权，抑制视盘、黄斑等高亮区域对细血管的干扰。
3. 输出与输入同分辨率，单通道 logits；阈值化后按任务要求做反相（血管 0，其他 255）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """用粗尺度特征生成门控图，对细尺度跳跃特征加权"""

    def __init__(self, gate_ch, skip_ch, mid_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, mid_ch, 1, bias=False),
                                 nn.BatchNorm2d(mid_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, mid_ch, 1, bias=False),
                                  nn.BatchNorm2d(mid_ch))
        self.psi = nn.Sequential(nn.Conv2d(mid_ch, 1, 1), nn.Sigmoid())

    def forward(self, g, s):
        g = F.interpolate(self.gate(g), size=s.shape[-2:], mode='bilinear', align_corners=False)
        a = self.psi(F.relu(g + self.skip(s)))
        return s * a


class UNetVessel(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, use_attention=True):
        super().__init__()
        c = base_ch
        self.use_attention = use_attention
        self.inc = DoubleConv(in_ch, c)
        self.down1 = nn.MaxPool2d(2)
        self.c1 = DoubleConv(c, c * 2)
        self.down2 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(c * 2, c * 4)
        self.down3 = nn.MaxPool2d(2)
        self.c3 = DoubleConv(c * 4, c * 8)
        self.down4 = nn.MaxPool2d(2)
        self.c4 = DoubleConv(c * 8, c * 16)

        self.up3 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.d3 = DoubleConv(c * 16, c * 8)
        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.d2 = DoubleConv(c * 8, c * 4)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.d1 = DoubleConv(c * 4, c * 2)
        self.up0 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.d0 = DoubleConv(c * 2, c)

        if use_attention:
            self.ag3 = AttentionGate(c * 16, c * 8, c * 4)
            self.ag2 = AttentionGate(c * 8, c * 4, c * 2)
            self.ag1 = AttentionGate(c * 4, c * 2, c)
        self.outc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.c1(self.down1(x0))
        x2 = self.c2(self.down2(x1))
        x3 = self.c3(self.down3(x2))
        x4 = self.c4(self.down4(x3))

        u3 = self.up3(x4)
        s3 = self.ag3(x4, x3) if self.use_attention else x3
        d3 = self.d3(torch.cat([u3, s3], dim=1))
        u2 = self.up2(d3)
        s2 = self.ag2(d3, x2) if self.use_attention else x2
        d2 = self.d2(torch.cat([u2, s2], dim=1))
        u1 = self.up1(d2)
        s1 = self.ag1(d2, x1) if self.use_attention else x1
        d1 = self.d1(torch.cat([u1, s1], dim=1))
        d0 = self.d0(torch.cat([self.up0(d1), x0], dim=1))
        return self.outc(d0)


if __name__ == '__main__':
    net = UNetVessel()
    print('输出:', tuple(net(torch.randn(1, 3, 512, 512)).shape))
    print('参数量: %.2f M' % (sum(p.numel() for p in net.parameters()) / 1e6))
