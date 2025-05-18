import cv2
import numpy as np
from PIL import Image
import os

def create_narrow_fov(image, ratio=3/5):
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

if __name__ == "__main__":

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


    