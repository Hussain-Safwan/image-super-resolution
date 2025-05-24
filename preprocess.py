import cv2
import numpy as np
from PIL import Image
import os
import sys
from utils import get_config, find_appr_dim

def create_narrow_fov(image, ratio):
    width, height = image.size
    new_width = int(width * ratio)  
    new_height = int(height * ratio)  
    
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    masked_image = Image.new("RGB", (width, height), (0, 0, 0))
    cropped_region = image.crop((left, top, right, bottom))
    masked_image.paste(cropped_region, (left, top))
    
    return masked_image

def downsize_wide_fov(image, factor=2):
    width, height = image.size
    width = int(width / factor)
    height = int(height / factor)
    resized_image = image.resize((width, height), Image.Resampling.BICUBIC)

    return resized_image

def simluate_single_image(image):
    w, h = image.size
    wide_w, wide_h = int(w/2), int(h/2)
    wide_w, wide_h = find_appr_dim(wide_w, 16), find_appr_dim(wide_h, 16)

    narrow_image = create_narrow_fov(image, 3/5)
    x, y = narrow_image.size
    narrow_w, narrow_h = find_appr_dim(x, 16), find_appr_dim(y, 16)

    wide_image = image.resize((wide_w, wide_h))
    narrow_image = narrow_image.resize(((narrow_w, narrow_h)))

    return wide_image, narrow_image

if __name__ == "__main__":

    narrow_ratio = get_config('narrow_ratio')
    single = sys.argv[1]

    if single:

        img_path = sys.argv[2]
        print(f'Preprocessing {img_path}...')
        image = Image.open(img_path)
        [filename, ext] = os.path.basename(img_path).split('.')
        wide, narrow = simluate_single_image(image)

        uploads_path = get_config('uploads_path')
        wide.save(f'{uploads_path}/{filename}_wide.{ext}')
        narrow.save(f'{uploads_path}/{filename}_narrow.{ext}')

        print(f'Processed narrow and wide images save at {uploads_path}')

    else:

        cnt = 1
        limit = 500
        home_dir = './dataset/lhq-kaggle/dataset'
        original_dir = './dataset/images/original'
        wide_dir = './dataset/images/wide'
        narrow_dir = './dataset/images/narrow'
        
        original_images = os.listdir(home_dir)

        for filename in original_images:
            img = Image.open(f'{home_dir}/{filename}')
            narrow_img = create_narrow_fov(img)
            wide_img = downsize_wide_fov(img)

            img.save(f'{original_dir}/{filename}')
            narrow_img.save(f'{narrow_dir}/{filename}')
            wide_img.save(f'{wide_dir}/{filename}')

            print(f'{cnt} - {filename}')
            cnt = cnt+1

            if (cnt > limit):
                break


    