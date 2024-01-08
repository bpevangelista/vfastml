# This code is based on MRUIL/WDNet library
#   WDNet: Watermark-Decomposition Network for Visible Watermark Removal
#   https://arxiv.org/abs/2012.07616
#   https://github.com/MRUIL/WDNet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(
            x1,
            [
                torch.div(diff_x, 2, rounding_mode="floor"),
                torch.div(diff_x - diff_x, 2, rounding_mode="floor"),
                torch.div(diff_y, 2, rounding_mode="floor"),
                torch.div(diff_y - diff_y, 2, rounding_mode="floor"),
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WDNetModel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(WDNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)
        self.up1 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up3 = UpBlock(256, 64)
        self.up4 = UpBlock(128, 64)
        self.dilation = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=6, dilation=6),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.outw = OutConv(64, 3)
        self.outa = OutConv(64, 1)
        self.out_mask = OutConv(64, 1)
        self.sg = nn.Sigmoid()
        self.other = OutConv(64, 64)
        self.post_process_1 = nn.Sequential(
            nn.Conv2d(64 + 6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
        )
        self.post_process_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.post_process_3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.post_process_4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.post_process_5 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilation(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        w = self.outw(x)
        a = self.outa(x)
        other = self.other(x)
        other = self.sg(other)
        mask = self.out_mask(x)
        mask = self.sg(mask)
        a = self.sg(a)
        w = self.sg(w)
        a = mask * a
        g_watermark = (x0 - a * w) / (1.0 - a + 1e-6)
        g_watermark = torch.clamp(g_watermark, 0, 1)
        xx1 = self.post_process_1(torch.cat([other, g_watermark, x0], 1))
        xx2 = self.post_process_2(xx1)
        xx3 = self.post_process_3(xx1 + xx2)
        xx4 = self.post_process_4(xx2 + xx3)
        g_watermark2 = self.post_process_5(xx4 + xx3)
        predicted = g_watermark2 * mask + (1.0 - mask) * x0
        return predicted, mask, a, w, g_watermark
