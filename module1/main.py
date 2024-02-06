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
image_path = "module1/photo140.png"
image = io.imread(image_path)

# Apply the yellow color mask as you did before
yellow_mask = image[:,:,0] > image[:,:,2]
yellow_mask &= image[:,:,1] > image[:,:,2]
yellow_mask &= np.abs(image[:,:,0] - image[:,:,1]) < 50

# Create a copy of the image
image_copy = np.copy(image)

# Convert the copy to grayscale
image_gray = color.rgb2gray(image_copy)

# Apply the mask
for i in range(3):
    image_copy[~yellow_mask, i] = image_gray[~yellow_mask] * 255

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

# Turn off empty subplot
# axs[0, 1].axis('off')

plt.tight_layout()
plt.show()
