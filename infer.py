from Models.Generator import get_generator
from utils import get_config, extract_patches, reconstruct, find_appr_dim
import tqdm
import sys
import os
import cv2
import datetime
from PIL import Image
from Dataloader.PatchSimilarity import PatchSimilarity

device = get_config('device')
patch_per_image = get_config("patch_per_image")

def get_similar_patches(wide_img, narrow_img):
    w, h = wide_img.size
    x, y = narrow_img.size

    wide_patch_w, wide_patch_h = int(w/patch_per_image), int(h/patch_per_image)
    narrow_patch_w, narrow_patch_h = int(x/patch_per_image), int(y/patch_per_image)

    wide_patches = extract_patches(wide_img, wide_patch_w, wide_patch_h)
    narrow_patches = extract_patches(narrow_img, narrow_patch_w, narrow_patch_h)

    similarity = PatchSimilarity(wide_patches, narrow_patches)
    patches = similarity.get_patches()

    return patches

def run_inference(generator, patches):
    wide_patches, narrow_patches, sr_patches = [], [], []

    for patch in patches:
        wide_patches.append(patch[0])
        narrow_patches.append(patch[1])

    tqdm_bar = tqdm.tqdm(
        total=len(patches),
        desc=f'Generating image',
        leave=False,
        position=0
    )

    for w, n in zip(narrow_patches, wide_patches):
        w, n = w.unsqueeze(0).to(device), n.unsqueeze(0).to(device)
        generated_patch = generator(w, n)
        generated_patch = generated_patch.detach().cpu().squeeze().clamp(0, 1).permute(1, 2, 0).numpy()
        sr_patches.append(generated_patch)
        tqdm_bar.update(1)

    tqdm_bar.close()
    return sr_patches

if __name__ == '__main__':
    gen_model_path = get_config('gen_cp_path')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_config('output_path') + f'{timestamp}.jpg'

    if gen_model_path == '' or not os.path.exists(gen_model_path):
        raise Exception('Generator model not found.')
    
    wide_filepath, narrow_filepath = sys.argv[1], sys.argv[2]
    print(f'Images loaded from {wide_filepath}, {narrow_filepath}')
    
    wide_img = Image.open(wide_filepath).convert('RGB')
    narrow_img = Image.open(narrow_filepath).convert('RGB')
    generator, _ = get_generator(gen_model_path)

    patches = get_similar_patches(wide_img, narrow_img)
    
    sr_patches = run_inference(generator, patches)
    sr_patches = [patch.transpose(1, 0, 2) for patch in sr_patches]

    width, height = wide_img.size
    base = wide_img.resize((width * 2, height * 2))

    patch_dim = (sr_patches[0].shape[0], sr_patches[0].shape[1])
    original_dim = (find_appr_dim(width*2, patch_dim[1]), find_appr_dim(height*2, patch_dim[0]))

    output = reconstruct(sr_patches, base, original_dim, patch_dim)

    cv2.imwrite(output_path, output)
    print(f'Image save at {output_path}')
    
    # python infer.py ./Dataset/wide/00005.jpg ./Dataset/narrow/00005.jpg
    # python infer.py ./Dataset/uploads/tween_wide.png ./Dataset/uploads/tween_narrow.png
    # python preprocess.py --single ./tween.png 

