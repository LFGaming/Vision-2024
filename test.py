##### Original Canvas Code #####
# from skimage import data
# from skimage.viewer import ImageViewer

# image = data.camera()
# viewer = ImageViewer(image)
# viewer.show()

##### Working Version of Canvas Code #####
# from skimage import data
# import matplotlib.pyplot as plt

# image = data.camera()
# plt.imshow(image, cmap='gray')
# plt.axis('off')  # turn off axis labels
# plt.show()

##### Show Picture in MatPlotLib #####
from skimage import io
import matplotlib.pyplot as plt

image_path = "module1/photo100.png"
image = io.imread(image_path)
plt.imshow(image, cmap='gray')
plt.axis('off')  # turn off axis labels
plt.show()