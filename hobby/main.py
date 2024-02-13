from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

def plot_hue_histogram(image, title, ax):
    # Convert the image to the HSV color space
    image_hsv = color.rgb2hsv(image)
    
    # Extract the Hue channel
    hue_values = image_hsv[:,:,0].flatten()
    
    # Plot the histogram of the Hue values
    ax.hist(hue_values, bins=180, color='orange', range=(0, 1))  # Hue values range from 0 to 1
    ax.set_title(title)
    ax.set_xlabel('Hue')
    ax.set_ylabel('Frequency')

# Load the original image
image_path = "hobby/img5.png"
image = io.imread(image_path)

# If the image has an alpha channel, remove it
if image.shape[2] == 4:
    image = image[:,:,:3]

# Convert the image to the HSV color space
image_hsv = color.rgb2hsv(image)
# Convert the image to the HSV color space
image_hsv = color.rgb2hsv(image)

# Create a mask for red and pinkish-red color
red_mask1 = image_hsv[:,:,0] < 0.02  # Adjust this value to exclude brownish colors
red_mask2 = image_hsv[:,:,0] > 0.8  # Hue for pinkish-red is around 0.9
red_mask = red_mask1 | red_mask2  # Combine the masks

# Create a copy of the image
image_copy = np.copy(image)

# Convert the copy to grayscale
image_gray = color.rgb2gray(image_copy)

# Apply the mask: keep the red and pinkish-red color in the original image, turn others into grayscale
for i in range(3):  # for each channel
    image_copy[~red_mask, i] = image_gray[~red_mask] * 255

# Save the modified image to a file
output_path = "hobby/img5_modified.png"
io.imsave(output_path, image_copy.astype(np.uint8))
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot original image and its histogram
axs[0, 0].imshow(image)
axs[0, 0].axis('off')
plot_hue_histogram(image, 'Histogram of Original Image', axs[0, 1])

# Plot modified image and its histogram
axs[1, 0].imshow(image_copy)
axs[1, 0].axis('off')
plot_hue_histogram(image_copy, 'Histogram of Modified Image', axs[1, 1])

plt.tight_layout()
plt.show()