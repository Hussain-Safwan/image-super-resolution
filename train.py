import torch
import torch.nn.functional as F
from Dataloader.Traindataset import TrainDataset
from Dataloader.CustomDataloader import CustomDataloader
from utils import transform, reconstruct, compute_boundary_map, save_checkpoint, get_config
import tqdm
import numpy as np
from Models.Generator import get_generator
from Models.Discriminator import get_discriminator
from Loss.generator_loss import generator_loss
from Loss.discriminator_loss import discriminator_loss
import warnings

warnings.filterwarnings('ignore')
device = get_config('device')
dataset_path = get_config('dataset_path')
gen_checkpoint_path = get_config('gen_cp_path')
disc_checkpoint_path = get_config('disc_cp_path')

patch_loss = []
image_loss_data = []

generator, g_optim = get_generator(gen_checkpoint_path)
discriminator, d_optim = get_discriminator(disc_checkpoint_path)

def train_patch_block(patch_idx, patch_pair, tqdm_bar):
  wide_patch, narrow_patch, original_patch = patch_pair
  wide_patch = wide_patch.unsqueeze(0).to(device)
  narrow_patch = narrow_patch.unsqueeze(0).to(device)
  original_patch = original_patch.unsqueeze(0).to(device)

  generated_patch = generator(wide_patch, narrow_patch)
  loss = generator_loss(generated_patch, original_patch, narrow_patch)

  loss.backward()

  return loss, generated_patch

def train_image_block(image_idx, image, tqdm_bar, update):
    image_loss = 0
    sr_patches = []

    for patch_idx, patch_trio in enumerate(image):
        loss, generated_patch = train_patch_block(patch_idx, patch_trio, tqdm_bar)
        image_loss += loss.item()
        sr_patches.append(generated_patch)

    if update:
      image_loss_data.append(image_loss/len(image))
      tqdm_bar.update(1)

    return loss, sr_patches

def generator_pass(image_idx, image, base_image, epoch, tqdm_bar, train_disc, factor=0.1):
  loss_adv = 0.0
  loss_content, sr_patches = train_image_block(image_idx, image, tqdm_bar, True)

  if train_disc:
    boundary_map = compute_boundary_map(1024, 1024)
    sr_patches = [
        patch.detach().cpu().squeeze().permute(1, 2, 0).numpy() for patch in sr_patches
    ]
    sr_image = reconstruct(sr_patches, base_image)
    sr_image = torch.from_numpy(sr_image).unsqueeze(0).permute(0, 3, 2, 1).to(device)
    boundary_map = boundary_map.unsqueeze(0).to(device)

    pred_fake = discriminator(sr_image, boundary_map)
    label_real = torch.ones_like(pred_fake)

    loss_adv = F.mse_loss(pred_fake, label_real)
    (factor * loss_adv).backward()

  g_optim.zero_grad()
  g_optim.step()

  return loss_content + (factor * loss_adv)

def discriminator_pass(image_idx, image, base_image, epoch, tqdm_bar):
  loss, sr_patches = train_image_block(image_idx, image, tqdm_bar, False)

  boundary_map = compute_boundary_map(1024, 1024)

  sr_patches = [
        patch.detach().cpu().squeeze().permute(1, 2, 0).numpy() for patch in sr_patches
  ]
  sr_image = reconstruct(sr_patches, base_image)
  sr_image = torch.from_numpy(sr_image).unsqueeze(0).permute(0, 3, 2, 1).to(device)
  base_image = torch.from_numpy(np.array(base_image)).unsqueeze(0).permute(0, 3, 2, 1).to(device)
  boundary_map = boundary_map.unsqueeze(0).to(device)

  loss_D = discriminator_loss(sr_image, base_image, boundary_map)

  d_optim.zero_grad()
  loss_D.backward()
  d_optim.step()

  return loss_D

def train_model(dataloader, n_epochs, train_disc=False):
    print(f'Total images: {len(dataloader)}, total iterations: {n_epochs*len(dataloader)}')
    for epoch in range(n_epochs):
      tqdm_bar = tqdm.tqdm(
        total=len(dataloader),
        desc=f'Epoch {epoch+1}',
        leave=False,
        position=0
      )

      for image_idx, (image, base_image) in enumerate(dataloader):
        g_loss, d_loss = 0, 0
        if train_disc:
          d_loss = discriminator_pass(image_idx, image, base_image, epoch, tqdm_bar)

        g_loss = generator_pass(image_idx, image, base_image, epoch, tqdm_bar, train_disc)

      state = (
         generator, 
         discriminator,
         g_loss,
         d_loss,
         g_optim,
         d_optim
      )
      save_checkpoint(epoch, state, train_disc)
      tqdm_bar.close()


if __name__ == '__main__':
    num_images = get_config('num_images')
    start = get_config('start_index')

    dataset = TrainDataset(
       dataset_path, 
       num_images, 
       start=start, 
       drop_ratio=0.0, 
       transform=transform
    
    )
    custom_loader = CustomDataloader(dataset)
    dataloader = custom_loader.getData()

    train_model(dataloader, 10, train_disc=True)