import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import transform
import math
import os
from PIL import Image
import numpy as np
from utils import get_config

device = get_config('device')

class TrainDataset(Dataset):
    def __init__(self, dataset_path, num_images, start=0, drop_ratio=0.0, transform=None):
        self.transform = transform
        self.start = start
        self.num_images = num_images
        self.device = device
        self.drop_ratio = drop_ratio
        self.patch_mappings = []

        self.wide_images = self.load_images(os.path.join(dataset_path, 'wide'), num_images)
        self.narrow_images = self.load_images(os.path.join(dataset_path, 'original'), num_images)
        self.original_images = self.load_images(os.path.join(dataset_path, 'original'), num_images)
        print(f'loading in the range: {self.start} to {self.start+self.num_images}')

        self.precompute()

    def load_images(self, image_path, num_images):
        images = []
        filenames = sorted(os.listdir(image_path))
        filenames = filenames[self.start:self.start+self.num_images+1]
        image_paths = [os.path.join(image_path, filename) for filename in filenames]

        for i, path in enumerate(image_paths):
            image = Image.open(path).convert('RGB')
            images.append(image)
            if i == self.num_images - 1:
                break
        print(f'{image_path}: loaded {len(images)} images')
        return images

    def extract_patches(self, image, patch_size, stride, ratio=1):
        patches = []
        h, w = image.size
        n_patches = math.floor(ratio*ratio*(h * w) // (patch_size * patch_size))
        maxlen = math.floor(n_patches - self.drop_ratio*n_patches)
        image = np.array(image)

        central_h = int(h * ratio)
        central_w = int(w * ratio)

        top = (h - central_h) // 2
        left = (w - central_w) // 2
        bottom = top + central_h
        right = left + central_w

        for i in range(top, bottom  , stride):
            if (len(patches) == maxlen):
              break
            for j in range(left, right  , stride):
                if (len(patches) == maxlen):
                  break
                patch = image[i:i+patch_size, j:j+patch_size, :]
                patch = Image.fromarray(patch)
                patch = self.transform(patch)
                patches.append(patch)
        return torch.stack(patches).to(self.device)

    def precompute(self):
        for i in range(self.num_images):
            wide_patches = self.extract_patches(self.wide_images[i], 128, 128)
            narrow_patches = self.extract_patches(self.narrow_images[i], 256, 256)
            original_patches = self.extract_patches(self.original_images[i], 256, 256)
            base_image = self.original_images[i]
            self.patch_mappings.append((wide_patches, narrow_patches, original_patches, base_image))
        print(f'Patches per image {len(self.patch_mappings[0][1])}')

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.patch_mappings[idx]