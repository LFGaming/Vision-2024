import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np

def show_images(image_list, title_list):
    num_images = len(image_list)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

    for i in range(num_images):
        axes[i].imshow(image_list[i], cmap='gray')
        axes[i].set_title(title_list[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Load an example image
image = io.imread('module1\photo100.png')

# Define the affine transformation parameters
rotation_angle = np.deg2rad(30)  # 30 degrees
translation_shift = (50, 20)  # Shift by (50, 20) pixels
stretch_scale = (0.5, 1.5)  # Stretch in x direction by 0.5, and in y direction by 1.5

# Perform rotation
rotation_matrix = transform.AffineTransform(rotation=rotation_angle)
rotated_image = transform.warp(image, rotation_matrix)

# Perform translation
translation_matrix = transform.AffineTransform(translation=translation_shift)
translated_image = transform.warp(image, translation_matrix)

# Perform stretching
stretch_matrix = transform.AffineTransform(scale=stretch_scale)
stretched_image = transform.warp(image, stretch_matrix)

# Show all images in one screen
show_images([image, rotated_image, translated_image, stretched_image],
            ['Original', 'Rotated', 'Translated', 'Stretched'])
