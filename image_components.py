from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("./content/fox.jpg")
imgarr = np.array(img)

R = imgarr[:, :, 0]
G = imgarr[:, :, 1]
B = imgarr[:, :, 2]

gray = np.mean(imgarr, axis=2).astype(np.uint8)
neg_gray = 255 - gray
threshold = 128
binary = (gray > threshold) * 255
binary_neg = 255 - binary

plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(R, cmap='Reds')
plt.title('Red')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(G, cmap='Greens')
plt.title('Green')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(B, cmap='Blues')
plt.title('Blue')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(neg_gray, cmap='gray')
plt.title('Gray Negative')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(binary, cmap='gray')
plt.title('Binary')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(binary_neg, cmap='gray')
plt.title('Binary Negative')
plt.axis('off')

plt.tight_layout()
plt.show()
