import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read the image
img = cv2.imread('./content/pokemon.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Crop image
cropped = img_gray[50:250, 550:750]
# Display Original, Gray, and Cropped Images
plt.figure(figsize=(8,6))
plt.subplot(2,2,1), plt.imshow(img_rgb), plt.title('Original Image')
plt.subplot(2,2,2), plt.imshow(img_gray, cmap='gray'), plt.title('Gray Image')
plt.subplot(2,2,3), plt.imshow(cropped, cmap='gray'), plt.title('Cropped Image')
plt.tight_layout()
plt.show()
#Mean and Standard Deviation
mean_val = np.mean(cropped)
std_val = np.std(cropped)
print(f"Mean (m): {mean_val:.4f}")
print(f"Standard Deviation (s): {std_val:.4f}")

# Create checkerboard images
checker1 = np.uint8(np.kron([[1,0]*4,[0,1]*4]*4, np.ones((20,20))) > 0.8)
checker2 = np.uint8(np.kron([[1,0]*8,[0,1]*8]*8, np.ones((10,10))) > 0.5)

# Display checkerboards
plt.figure(figsize=(6,4))
plt.subplot(2,1,1), plt.imshow(checker1, cmap='gray'), plt.title('Image1')
plt.subplot(2,1,2), plt.imshow(checker2, cmap='gray'), plt.title('Image2')
plt.tight_layout()
plt.show()

# Correlation Coefficient
r = np.corrcoef(checker1.flatten(), checker2.flatten())[0,1]
print(f"Correlation Coefficient (r): {r:.4f}")