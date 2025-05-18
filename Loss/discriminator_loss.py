import torch
import torch.nn.functional as F

def discriminator_loss(image, gt, mask, discriminator):
    pred_fake = discriminator(image, mask)
    fake_labels = torch.zeros_like(pred_fake)
    loss_fake = F.mse_loss(pred_fake, fake_labels)

    pred_real = discriminator(gt, mask)
    real_labels = torch.ones_like(pred_real)
    loss_real = F.mse_loss(pred_real, real_labels)

    loss_total = (loss_fake + loss_real) / 2.0
    return loss_total