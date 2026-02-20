from skimage import io, color, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

img_path = "./content/landscape.jpg"

# Read image and convert to grayscale
I = io.imread(img_path)
if I.ndim == 3:
    I=color.rgb2gray(I)
I = img_as_ubyte(I) # convert to 8-bit grayscale

# Extract bit planes
bit_planes = [(I >> k) & 1 for k in range(8)] # bit 0 to bit 7

# Display original + 8 bit planes
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs = axs.ravel()

axs[0].imshow(I, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

for k in range(8):
    axs[k+1].imshow(bit_planes[k], cmap='gray')
    axs[k+1].set_title(f"BIT PLANE {k}")
    axs[k+1].axis('off')

plt.tight_layout(pad=2)
plt.show()