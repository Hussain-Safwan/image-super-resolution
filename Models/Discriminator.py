import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init 
from utils import get_config

device = get_config('device')

class Discriminator(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(base_channels, base_channels * 2, stride=2),
            conv_block(base_channels * 2, base_channels * 4, stride=2),
            conv_block(base_channels * 4, base_channels * 8, stride=1),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x_img, x_mask):
        x = torch.cat([x_img, x_mask], dim=1)
        return self.model(x)
    
def get_discriminator(filename):
  cp_found = False
  discriminator = Discriminator()
  discriminator.to(device)
  discriminator.apply(weights_init)
  optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=1e-5
  )

  if (os.path.exists(filename)):
    cp_found = True
    checkpoint = torch.load(filename, map_location=device)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  print(f'Discriminator checkpoint file: {"FOUND" if cp_found else "NOT_FOUND"}')
  return discriminator, optimizer