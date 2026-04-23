import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color =cv2.imread('./content/tiger.jpg')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
noise_img = img_gray.copy()
prob = 0.02

# Salt noise
salt = np.random.rand(*img_gray.shape) < prob
noise_img[salt] = 255
# Pepper noise
pepper = np.random.rand(*img_gray.shape) < prob
noise_img[pepper] = 0

median_3 = cv2.medianBlur(noise_img, 3)
median_9 = cv2.medianBlur(noise_img, 9)
mean_3 = cv2.blur(img_gray, (3,3))
mean_10 = cv2.blur(img_gray, (10,10))
kernel_3 = np.ones((3,3), np.float32) / 9
kernel_10 = np.ones((10,10), np.float32) / 100
filtered_3 = cv2.filter2D(img_gray, -1, kernel_3)
filtered_10 = cv2.filter2D(img_gray, -1, kernel_10)

# -------- FIRST OUTPUT (Median Section) --------
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.imshow(img_color)
plt.title("Original Image")
plt.axis('on')
plt.subplot(3,2,2)
plt.imshow(img_gray, cmap='gray')
plt.title("Gray Image")
plt.axis('on')
plt.subplot(3,2,3)
plt.imshow(noise_img, cmap='gray')
plt.title("Noise Added Image")
plt.axis('on')
plt.subplot(3,2,4)
plt.imshow(median_3, cmap='gray')
plt.title("3x3 Median Filter")
plt.axis('on')
plt.subplot(3,2,5)
plt.imshow(median_9, cmap='gray')
plt.title("9x9 Median Filter")
plt.axis('on')
plt.tight_layout()
plt.show()

# -------- SECOND OUTPUT (Mean Section) --------
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img_color)
plt.title("Original Image")
plt.axis('on')
plt.subplot(2,2,2)
plt.imshow(img_gray, cmap='gray')
plt.title("Gray Image")
plt.axis('on')
plt.subplot(2,2,3)
plt.imshow(mean_3, cmap='gray')
plt.title("3x3 Mean Filter")
plt.axis('on')
plt.subplot(2,2,4)
plt.imshow(mean_10, cmap='gray')
plt.title("10x10 Mean Filter")
plt.axis('on')
plt.tight_layout()
plt.show()

# -------- THIRD OUTPUT (Convolution Section) --------
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original Image")
plt.axis('on')
plt.subplot(2,2,2)
plt.imshow(filtered_3, cmap='gray')
plt.title("Filtered Image 1 (3x3)")
plt.axis('on')
plt.subplot(2,2,3)
plt.imshow(filtered_10, cmap='gray')
plt.title("Filtered Image 2 (10x10)")
plt.axis('on')
plt.tight_layout()
plt.show()