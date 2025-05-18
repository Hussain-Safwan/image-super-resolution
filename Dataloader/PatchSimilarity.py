import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from utils import get_config

device = get_config('device')
vgg = models.vgg19(pretrained=True).features.to(device).eval()

class PatchSimilarity:
  def __init__(self, wide, narrow):
        self.wide_patches = wide
        self.narrow_patches = narrow
        self.vgg = self.get_vgg19_features()
        self.set_min_max()
        self.compute_similarity()

  def get_vgg19_features(self):
        return torch.nn.Sequential(*list(vgg.children())[:9])

  def resize(self, tensor, width, height):
        return transforms.Resize((height, width))(tensor)

  def set_min_max(self):
        self.vgg = self.vgg.to(device)
        patch1 = torch.randn(1, 3, 128, 128).to(device)
        patch2 = torch.zeros(1, 3, 128, 128).to(device)
        with torch.no_grad():
            patch1_features = self.vgg(patch1).view(1, -1)
            patch2_features = self.vgg(patch2).view(1, -1)

        patch1_features = F.normalize(patch1_features, dim=1)
        patch2_features = F.normalize(patch2_features, dim=1)

        distance = torch.norm(patch1_features - patch2_features).item()

        self.max_diff = distance

  def compute_similarity(self):
        self.vgg = self.vgg.to(device)

        original_narrow_patches = self.narrow_patches

        if self.wide_patches[0].shape[1] < self.narrow_patches[0].shape[1]:
            _, min_w, min_h = self.wide_patches[0].shape
            self.narrow_patches = torch.stack([self.resize(patch, min_w, min_h) for patch in self.narrow_patches])

        with torch.no_grad():
            wide_features = self.vgg(self.wide_patches).view(self.wide_patches.size()[0], -1)
            narrow_features = self.vgg(self.narrow_patches).view(self.narrow_patches.size()[0], -1)

        wide_features = F.normalize(wide_features, dim=1)
        narrow_features = F.normalize(narrow_features, dim=1)
        n=wide_features.size()[0]

        distances = torch.cdist(wide_features, narrow_features)
        min_indices = torch.argmin(distances, dim=1)

        self.matched_narrow_patches = original_narrow_patches[min_indices]
        min_distances = torch.gather(distances, 1, min_indices.unsqueeze(1))
        self.min_distances = min_distances.detach().cpu().apply_(lambda x: 1 - x/self.max_diff)

  def get_patches(self):
        patches = []

        for i in range(len(self.wide_patches)):
            patch = (
                self.wide_patches[i],
                self.matched_narrow_patches[i],
                self.min_distances[i].item()
            )
            patches.append(patch)
        return patches