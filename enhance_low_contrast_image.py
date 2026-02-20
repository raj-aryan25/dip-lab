from skimage import io, color, exposure, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

img_path = "./content/fox.jpg"

# Read image
I= io.imread(img_path)

# If image has alpha channel, drop it
if I.ndim == 3 and I.shape[2] == 4:
    I = I[ ... , :3]

# Normalize to float [0,1]
if I.dtype not in [np.float32, np.float64]:
    I_float = (I /255.0).astype(np.float32)
else:
    I_float = I.copy()

# Convert to grayscale (float [0,1])
g = color.rgb2gray(I_float)

# Contrast stretching similar to MATLAB imadjust(g,[0.3 0.7], [])
in_min, in_max = 0.3, 0.7
J= exposure.rescale_intensity(g, in_range=(in_min, in_max), out_range=(0, 1))

# False-color enhanced image: map grayscale through a vivid colormap
gray_for_colormap = img_as_ubyte(g)
cmap = plt.get_cmap('inferno') # try 'viridis','plasma','inferno' etc.
D_color = cmap(gray_for_colormap / 255.0)[ ... , :3] # drop alpha

# Histogram equalization of grayscale
m = exposure.equalize_hist(g)

# Plot similar layout (4 rows x 2 cols)
fig, axs = plt.subplots(4, 2, figsize=(10,13))
plt.tight_layout(pad=3.0)

axs[0, 0].imshow(I_float)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(D_color)
axs[0, 1].set_title("Enhanced Image 2 (False Color)")
axs[0, 1].axis('off')

axs[1, 0].imshow(J, cmap='gray', vmin=0, vmax=1)
axs[1, 0].set_title("Enhanced Image (contrast stretch)")
axs[1, 0].axis('off')

axs[1, 1].imshow(m, cmap='gray', vmin=0, vmax=1)
axs[1, 1].set_title("Equalized Image")
axs[1, 1].axis('off')

axs[2, 0].imshow(g, cmap='gray', vmin=0, vmax=1)
axs[2, 0].set_title("Gray Image")
axs[2, 0].axis('off')

# Histogram of equalized image
axs[2, 1].hist((m.ravel()*255).astype(np.uint8), bins=256, range=(0,255))
axs[2, 1].set_title("Histogram of Equalized Image")
axs[2, 1].set_xlim(0,255)

# Histogram of original gray image
axs[3, 0].hist((g.ravel()*255).astype(np.uint8), bins=256, range=(0,255))
axs[3, 0].set_title("Histogram of Gray Image")
axs[3, 0].set_xlim(0,255)

# small original preview for balance
axs[3, 1].imshow(I_float, extent=[0,1,0,1])
axs[3, 1].set_title("Original (small)")
axs[3, 1].axis('off')

plt.subplots_adjust(hspace=0.6)
plt.show()