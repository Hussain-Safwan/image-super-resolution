from Models.Generator import get_generator
from utils import get_config, extract_patches, reconstruct
import sys
import os
import cv2
import datetime
from PIL import Image
from Dataloader.PatchSimilarity import PatchSimilarity

device = get_config('device')

def run_inference(generator, patches):
    wide_patches, narrow_patches, sr_patches = [], [], []

    for patch in patches:
        wide_patches.append(patch[0])
        narrow_patches.append(patch[1])

    for w, n in zip(narrow_patches, wide_patches):
        w, n = w.unsqueeze(0).to(device), n.unsqueeze(0).to(device)
        generated_patch = generator(w, n)
        generated_patch = generated_patch.detach().cpu().squeeze().clamp(0, 1).permute(1, 2, 0).numpy()
        sr_patches.append(generated_patch)

    return sr_patches

if __name__ == '__main__':
    gen_model_path = get_config('gen_cp_path')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_config('output_path') + f'{timestamp}.jpg'

    if gen_model_path == '' or not os.path.exists(gen_model_path):
        raise Exception('Generator model not found.')
    
    wide_filepath, narrow_filepath = sys.argv[1], sys.argv[2]
    wide_img = Image.open(wide_filepath).convert('RGB')
    narrow_img = Image.open(narrow_filepath).convert('RGB')
    generator, _ = get_generator(gen_model_path)

    wide_dim, narrow_dim = wide_img.size[0], narrow_img.size[0]
    wide_patch_size, narrow_patch_size = int(wide_dim/16), int(narrow_dim/16)

    wide_patches = extract_patches(wide_img, wide_patch_size, wide_patch_size)
    narrow_patches = extract_patches(narrow_img, narrow_patch_size, narrow_patch_size)

    similarity = PatchSimilarity(wide_patches, narrow_patches)
    patches = similarity.get_patches()
    
    sr_patches = run_inference(generator, patches)

    width, height = wide_img.size
    base = wide_img.resize((width * 2, height * 2))

    patch_dim = sr_patches[0].shape[0]
    original_dim = patch_dim * 16

    output = reconstruct(sr_patches, base, original_dim, patch_dim)

    cv2.imwrite(output_path, output)
    print(f'Image save at {output_path}')
    
    # python infer.py ./Dataset/wide/00005.jpg ./Dataset/narrow/00005.jpg
    # python infer.py ./Dataset/uploads/me_wide.jpg ./Dataset/uploads/me_narrow.jpg

