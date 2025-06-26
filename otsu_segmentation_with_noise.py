import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create an empty grayscale image (100x100) - background value 0
image = np.zeros((100, 100), dtype=np.uint8)

#Add Object 1 - a gray rectangle with pixel value 100
image[20:50, 20:50] = 100

#Add Object 2 - a lighter gray circle with pixel value 200
cv2.circle(image, (70, 70), 15, 200, -1)

#Gaussian noise-mean=0,std=10
noise = np.random.normal(0, 10, image.shape)
noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

#Apply Otsu’s thresholding
_, otsu_thresh = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot and save
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Otsu Thresholded")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("otsu_segmentation_output.jpg")
# Save the original grayscale image
cv2.imwrite("noisy_image.jpg", noisy_image)

print("Image saved as otsu_segmentation_output.jpg")
plt.show()
