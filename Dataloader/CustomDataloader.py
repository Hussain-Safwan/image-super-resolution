from Dataloader.PatchSimilarity import PatchSimilarity
import tqdm

class CustomDataloader:
  def __init__(self, dataset):
    self.dataset = dataset
    self.size = dataset.__len__()

  def getData(self, num_data=None):
    if num_data == None or num_data > self.size:
      num_data = self.size
      progress = tqdm(
          total=num_data,
          desc='Loading data',
          leave=False,
          position=0
      )

    data = []
    for i in range(num_data):
      wide_patches, narrow_patches, original_patches, base_image = self.dataset.__getitem__(i)
      similarity = PatchSimilarity(wide_patches, narrow_patches)
      patches = similarity.get_patches()
      patches = [(patch[0], patch[1], original_patch) for patch, original_patch in zip(patches, original_patches)]
      data.append((patches, base_image))
      progress.update(1)

    progress.close()
    return data