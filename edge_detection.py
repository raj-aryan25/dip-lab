import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./content/pokemon.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F, ksize=3)
sharp = np.clip(gray - lap, 0, 255).astype(np.uint8)

sx = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
edge_sobel = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))

kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
gx = cv2.filter2D(gray.astype(np.uint8), cv2.CV_64F, kx)
gy = cv2.filter2D(gray.astype(np.uint8), cv2.CV_64F, ky)
edge_prewitt = np.uint8(np.clip(np.sqrt(gx**2 + gy**2), 0, 255))

kx = np.array([[0, 1], [-1, 0]], dtype=np.float64)
ky = np.array([[1, 0], [0, -1]], dtype=np.float64)
gx = cv2.filter2D(gray.astype(np.uint8), cv2.CV_64F, kx)
gy = cv2.filter2D(gray.astype(np.uint8), cv2.CV_64F, ky)
edge_roberts = np.uint8(np.clip(np.sqrt(gx**2 + gy**2), 0, 255))

lap_edges = np.uint8(np.clip(np.abs(lap), 0, 255))
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(gray.astype(np.uint8), cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(sharp, cmap='gray')
ax[0, 1].set_title('Sharpened (Laplacian)')
ax[0, 1].axis('off')
ax[0, 2].imshow(lap_edges, cmap='gray')
ax[0, 2].set_title('Laplacian Edges')
ax[0, 2].axis('off')
ax[1, 0].imshow(edge_sobel, cmap='gray')
ax[1, 0].set_title('Sobel')
ax[1, 0].axis('off')
ax[1, 1].imshow(edge_prewitt, cmap='gray')
ax[1, 1].set_title('Prewitt')
ax[1, 1].axis('off')
ax[1, 2].imshow(edge_roberts, cmap='gray')
ax[1, 2].set_title('Roberts')
ax[1, 2].axis('off')
plt.tight_layout()
plt.show()