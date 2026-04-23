import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('./content/tiger.jpg')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

cropped_img = img_gray[0:200, 0:200]
mean_val = np.mean(cropped_img)
std_val = np.std(cropped_img)
size = 200
block = 20

image1 = np.zeros((size, size))
for i in range(0, size, block):
    for j in range(0, size, block):
        if (i//block + j//block) % 2 == 0:
            image1[i:i+block, j:j+block] = 255

block2 = 10
image2 = np.zeros((size, size))
for i in range(0, size, block2):
    for j in range(0, size, block2):
        if (i//block2 + j//block2) % 2 == 0:
            image2[i:i+block2, j:j+block2] = 255

corr = np.corrcoef(image1.flatten(), image2.flatten())[0,1]
plt.figure(figsize=(10,10))
# Original Image
plt.subplot(3,2,1)
plt.imshow(img_color)
plt.title("Original Image")
plt.axis('on')

# Gray Image
plt.subplot(3,2,2)
plt.imshow(img_gray, cmap='gray')
plt.title("Gray Image")
plt.axis('on')

# Cropped Image
plt.subplot(3,2,3)
plt.imshow(cropped_img, cmap='gray')
plt.title("Cropped Image")
plt.axis('on')

# Image1
plt.subplot(3,2,5)
plt.imshow(image1, cmap='gray')
plt.title("Image1")
plt.axis('on')

# Image2
plt.subplot(3,2,6)
plt.imshow(image2, cmap='gray')
plt.title("Image2")
plt.axis('on')
plt.tight_layout()
plt.show()

# Print Mean, Std, Correlation
print("Mean (m):", round(mean_val,4))
print("Standard Deviation (s):", round(std_val,4))
print("Correlation Coefficient (r):", round(corr,4))