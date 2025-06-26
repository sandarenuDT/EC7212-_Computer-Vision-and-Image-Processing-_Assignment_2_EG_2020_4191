import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed, threshold=5):
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)

    seed_value = int(image[seed])
    region_mean = seed_value
    region_size = 1

    # Stack for region growing (x, y) coordinates
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue

        visited[x, y] = True
        pixel_value = int(image[x, y])

        if abs(pixel_value - region_mean) <= threshold:
            segmented[x, y] = 255
            region_mean = (region_mean * region_size + pixel_value) / (region_size + 1)
            region_size += 1

            # 4-neighbors (up, down, left, right)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    stack.append((nx, ny))

    return segmented

# Load grayscale image
image = cv2.imread('otsu_segmentation_output.jpg', cv2.IMREAD_GRAYSCALE)

# Set seed point manually (e.g., middle of image)
seed_point = (int(image.shape[0]/2), int(image.shape[1]/2))

# Apply region growing
segmented = region_growing(image, seed_point, threshold=20)

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Segmented Region')
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("region_growing_segmentation_output.jpg")
print("Image saved as region_growing_segmentation_output.jpg")
plt.show()
