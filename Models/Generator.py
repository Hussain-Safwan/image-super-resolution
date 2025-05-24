import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init 
from utils import get_config

device = get_config('device')

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = math.sqrt(channels)
        self.down = nn.Upsample(scale_factor=0.25, mode='bilinear')

    def forward(self, feat_wide, feat_narrow):
        B, C, H, W = feat_wide.shape

        w_ds = self.down(feat_wide)  # e.g. 32×32
        n_ds = self.down(feat_narrow)

        Q = w_ds.view(B, C, -1).permute(0, 2, 1)
        K = n_ds.view(B, C, -1)
        V = K

        attn = torch.softmax(Q @ K / self.scale, dim=-1)
        out = attn @ V.transpose(1, 2)
        out = out.permute(0, 2, 1).view(B, C, w_ds.size(2), w_ds.size(3))
        out = F.interpolate(out, size=(H, W), mode='bilinear')

        del Q, K, V
        return feat_wide + out


class AttentionResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attn = AttentionFusion(channels)

    def forward(self, wide_feat, narrow_feat):
        out = self.relu(self.conv1(wide_feat))
        out = self.attn(out, narrow_feat)
        out = self.conv2(out)
        return out + wide_feat


class AttentiveGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.resblocks = nn.ModuleList([
            AttentionResBlock(64) for _ in range(num_blocks)
        ])

        self.exit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, wide_patch, narrow_patch):
        x = self.entry(wide_patch)
        y = self.entry(narrow_patch)

        for block in self.resblocks:
            x = block(x, y)
        return self.exit(x)
    
def get_generator(filename):
  cp_found = False
  generator = AttentiveGenerator()
  generator.to(device)
  generator.apply(weights_init)
  optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=1e-5
  )

  if (os.path.exists(filename)):
    cp_found = True
    checkpoint = torch.load(filename, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  print(f'{f"Generator checkpoint loaded from {filename}" if cp_found else "Generator model not found. Proceeding with initialized weights."}')
  
  return generator, optimizer