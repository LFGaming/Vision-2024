from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

# Load the image
image = data.camera()

# Display the original image
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')  # turn off axis labels

# Define the convolution kernel
kernel = np.array([[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]])

# Apply convolution filter
new_image = scipy.ndimage.convolve(image, kernel)
new_image = scipy.ndimage.convolve(new_image, kernel)

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(new_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')  # turn off axis labels

plt.show()
