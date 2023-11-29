import imgaug.augmenters as iaa
import cv2
import numpy as np
import os

augmentation = iaa.Sequential([
    iaa.Fliplr(0.4),   # Horizontal flip with a 40% probability
    iaa.Flipud(0.2),   # Vertical flip with a 20% probability
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # Apply Gaussian blur
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),  # Adjust contrast
    iaa.Affine(
        rotate=(-30, 30),  # Rotate images between -30 and 30 degrees
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate along x and y axes
        scale=(0.8, 1.2),  # Scale images
        shear=(-16, 16)  # Shear images
    )
])

input_dir = './data'
output_dir = './augmented_data'

batch_size = 10
for image_filename in os.listdir(input_dir):
    image = cv2.imread(os.path.join(input_dir, image_filename))
    for i in range(batch_size): 
        image_augmented = augmentation(image=image)
        augmented_filename = os.path.join(output_dir, f"{image_filename[:-4]}_augmented{i}.jpg")
        cv2.imwrite(augmented_filename, image_augmented)