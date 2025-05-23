import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torchvision.transforms as transforms
import numpy as np
import json
import math
from PIL import Image

def get_config(key):
   with open('config.json') as f:
      config = json.load(f)
      if key in config:
        return config[key]
      else:
         raise Exception(f'Item {key} not found in config.json')
   
device = get_config('device')

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)

transform = transforms.Compose([
    transforms.ToTensor(),
])
      
def compute_boundary_map(height, width, patch_size=256, thickness=1):
    mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)

    for x in range(patch_size, width, patch_size):
        mask[:, :, max(0, x - thickness // 2):min(width, x + thickness // 2 + 1)] = 1.0

    for y in range(patch_size, height, patch_size):
        mask[:, max(0, y - thickness // 2):min(height, y + thickness // 2 + 1), :] = 1.0

    return mask

def extract_patches(image, patch_size_w, patch_size_h, drop_ratio=0.0, ratio=1):
        patches = []
        h, w = image.size
        n_patches = math.floor(ratio*ratio*(h * w) // (patch_size_w * patch_size_h))
        maxlen = math.floor(n_patches - drop_ratio*n_patches)
        image = np.array(image)

        central_h = int(h * ratio)
        central_w = int(w * ratio)

        top = (h - central_h) // 2
        left = (w - central_w) // 2
        bottom = top + central_h
        right = left + central_w

        row=1
        for i in range(top, bottom, patch_size_w):
            row = row+1
            if (len(patches) == maxlen):
              break
            for j in range(left, right, patch_size_h):
                if (len(patches) == maxlen):
                  break

                patch = image[i:i+patch_size_w, j:j+patch_size_h, :]
                patch = Image.fromarray(patch)
                patch = transform(patch)
                if len(patches) == 0 or patches[0].shape == patch.shape:
                  patches.append(patch)
        return torch.stack(patches).to(device)

def reconstruct(patches, base, original_dim, patch_dim):
  res = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
  res = cv2.resize(res, (original_dim[0], original_dim[1]), interpolation=cv2.INTER_AREA)

  n_width = original_dim[0] // patch_dim[1]
  n_height = original_dim[1] // patch_dim[0]
  patch_no = 0

  for i in range(n_width):
    for j in range(n_height):

      x = i * patch_dim[0]
      y = j * patch_dim[1]
      patch = patches[patch_no]
      patch = (patch * 255).astype(np.uint8)

      patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
      mask = np.ones(patch_dim, dtype=np.uint8) * 255
      center = (y + patch_dim[1] // 2, x + patch_dim[0] // 2)
      res = cv2.seamlessClone(patch, res, mask, center, cv2.MIXED_CLONE)
      patch_no += 1

  return res

def find_appr_dim(dim, n):
  lower = (dim // n) * n
  upper = lower + n

  if dim - lower < upper - dim:
    return lower
  else:
    return upper

def save_checkpoint(epoch, state, train_disc):
  generator, discriminator, g_loss, d_loss, g_optim, d_optim = state[0], state[1], state[2], state[3], state[4], state[5]
  
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  suffix = f'ep_{epoch+1}-{timestamp}.pth'
  gen_checkpoint = {
    'model_state_dict': generator.state_dict(),
    'optimizer_state_dict': g_optim.state_dict(),
  }
  torch.save(gen_checkpoint, f'gen-{suffix}')

  if train_disc:
    dis_checkpoint = {
      'model_state_dict': discriminator.state_dict(),
      'optimizer_state_dict': d_optim.state_dict(),
    }
    torch.save(dis_checkpoint, f'disc-{suffix}')

  print(f'''
        Epoch {epoch+1} |
        G: {g_loss:.3f}, {f"D: {d_loss:.3f}" if train_disc else ""} |
        Checkpoints: {f"gen_{suffix}"}, {f"disc_{suffix}" if train_disc else ""}
  ''')

