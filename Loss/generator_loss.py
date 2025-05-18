import torch
import torchvision.models as models
import torch.nn.functional as F
from utils import get_config

device = get_config('device')

vgg = models.vgg19(pretrained=True).features.to(device).eval()

def get_features_using_vgg_layers(tensor):
    layers = {
        'conv1_1': vgg[0],   # First conv layer
        'conv2_1': vgg[5],   # First conv layer of second block
        'conv3_1': vgg[10],  # First conv layer of third block
        'conv4_1': vgg[19],  # First conv layer of fourth block
        'conv5_1': vgg[28],  # First conv layer of fifth block
    }

    for layer in layers:
        tensor = layers[layer](tensor)
    return tensor

def compute_image_properties(tensor):
    mean_color = torch.mean(tensor, dim=[-2, -1], keepdim=True)
    brightness = torch.mean(tensor)
    contrast = torch.std(tensor)
    laplacian = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False).to(device)
    laplacian.weight.data = torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]).expand(3, 3, 3, 3).to(device)
    sharpness = torch.mean(torch.abs(laplacian(tensor)))

    return mean_color, brightness, contrast, sharpness

def augmented_gram_matrix(tensor):
    features = get_features_using_vgg_layers(tensor)
    _, c, h, w = features.size()
    features = features.view(c, h * w)
    gram = torch.matmul(features, features.t()) / (c * h * w)
    mean_color, brightness, contrast, sharpness = compute_image_properties(tensor)
    augmentation = mean_color.mean() + brightness + contrast + sharpness

    return gram * augmentation

def content_loss(patch1, patch2):
    return F.smooth_l1_loss(patch1, patch2)

def visual_loss(patch1, patch2):
    gram1 = augmented_gram_matrix(patch1)
    gram2 = augmented_gram_matrix(patch2)
    gram1 = torch.clamp(gram1.type(torch.float32), min=-1e12, max=1e12)
    gram2 = torch.clamp(gram2.type(torch.float32), min=-1e12, max=1e12)

    return F.smooth_l1_loss(gram1, gram2)

def generator_loss(generated_patches, narrow_patches, wide_patches, alpha=1.0, beta=1.0):
    c_loss = content_loss(generated_patches, wide_patches)
    v_loss = visual_loss(generated_patches, narrow_patches)

    return alpha * c_loss + beta * v_loss