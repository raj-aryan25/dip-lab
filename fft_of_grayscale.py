from skimage import io, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt

img_path = "./content/landscape.jpg"

# Read and convert image to grayscale float
I = io.imread(img_path)
if I.ndim == 3:
    I= color.rgb2gray(I)
    I = img_as_float(I)

# 1D FFT
f1= np.fft.fft(I)       # 1D FFT along rows
f2= np.fft.fftshift(f1) # Shift zero frequency to center

# ---- Visualization
plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.imshow(np.abs(f1), cmap='gray')
plt.title('Frequency Spectrum')
plt.axis('off')

plt.subplot(2, 2,2)
plt.imshow(np.abs(f2), cmap='gray')
plt.title('Centered Spectrum')
plt.axis('off')

# ---- Log spectrum for better visibility
f3=np.log(1+np.abs(f2))
plt.subplot(2,2,3)
plt.imshow(f3, cmap='gray')
plt.title('log(1+abs(f2))')
plt.axis('off')

# --- 2D FFT 
f_2d= np.fft.fft2(I)
I1= np.real(np.fft.ifft2(f_2d))  # Inverse FFT to reconstruct image
plt.subplot(2, 2, 4)
plt.imshow(I1, cmap='gray')
plt.title('2-D FFT')
plt.axis('off')

plt.tight_layout(pad=2)
plt.show()

