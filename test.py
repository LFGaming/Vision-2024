from skimage import data
import matplotlib.pyplot as plt

image = data.camera()
plt.imshow(image, cmap='gray')
plt.axis('off')  # turn off axis labels
plt.show()

# from skimage import data
# from skimage.viewer import ImageViewer

# image = data.camera()
# viewer = ImageViewer(image)
# viewer.show()