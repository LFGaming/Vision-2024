from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

# Load the image
image = data.camera()

# Display the original image
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Define the convolution kernels
blur = np.array([[1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9]])

kernel1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

kernel1_2 = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

kernel2 = np.array([[0.5, 1, 0.5],
                    [1, -6, 1],
                    [0.5, 1, 0.5]])

kernel3 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

# Apply convolution filters step by step
new_image1 = scipy.ndimage.convolve(image, blur)
new_image1 = scipy.ndimage.convolve(new_image1, blur)
# new_image1 = scipy.ndimage.convolve(new_image1, blur)

new_image1_2 = scipy.ndimage.convolve(new_image1, kernel1)
new_image1_3 = scipy.ndimage.convolve(new_image1, kernel1_2)

kn1 = np.power(new_image1_2, 2)
kn2 = np.power(new_image1_3, 2)
kn1_2 = np.sqrt(kn1 + kn2)

new_image1 = kn1_2

new_image2 = scipy.ndimage.convolve(image, blur)
new_image2 = scipy.ndimage.convolve(new_image2, kernel2)

new_image3 = scipy.ndimage.convolve(image, blur)
new_image3 = scipy.ndimage.convolve(new_image3, kernel3)

# Display the filtered images
plt.subplot(2, 2, 2)
plt.imshow(new_image1+127, cmap='gray')
plt.title('Filtered Image 1')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(new_image2+127, cmap='gray')
plt.title('Filtered Image 2')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(new_image3+127, cmap='gray')
plt.title('Filtered Image 3')
plt.axis('off')

plt.show()
