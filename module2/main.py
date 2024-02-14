from skimage import data, filters, feature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# Load the image
image = data.camera()

# Display the original image
plt.figure(figsize=(18, 12))
plt.subplot(3, 3, 1)
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
                    [0.5, 1, 0.5]]) # Laplacian diagonals

kernel3 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]) # High-Pass

kernel4 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]]) # Laplacian up and down added in one

# Apply convolution filters step by step
new_image1 = filters.gaussian(image)

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

new_image4 = scipy.ndimage.convolve(image, blur)
new_image4 = scipy.ndimage.convolve(new_image4, kernel4)

# Display the filtered images
plt.subplot(3, 3, 2)
plt.imshow(new_image1+127, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(new_image2+127, cmap='gray')
plt.title('Laplacian Diagonals')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(new_image3+127, cmap='gray')
plt.title('High-pass filtering')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(new_image4 + 127, cmap='gray')
plt.title('Laplacian Added Hor and Ver')
plt.axis('off')

# Adding more edge detection functions
new_image5 = filters.prewitt(image)
new_image6 = filters.sobel(image)
new_image7 = filters.roberts(image)
new_image8 = feature.canny(image, sigma=2)

plt.subplot(3, 3, 7)
plt.imshow(new_image5, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(new_image6, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(new_image7, cmap='gray')
plt.title('Roberts Edge Detection')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(new_image8, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
