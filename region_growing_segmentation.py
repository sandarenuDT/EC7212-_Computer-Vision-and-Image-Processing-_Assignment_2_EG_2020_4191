import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_grow_single(image, seed, threshold=20):
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)

    seed_value = int(image[seed])
    region_mean = seed_value
    region_size = 1

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

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    stack.append((nx, ny))

    return segmented

# Load grayscale image
image = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

# Seed points: one inside square, one inside circle
seed_points = [(34, 33), (67, 71)]

# Apply region growing independently for each seed
segmented_total = np.zeros_like(image)
for seed in seed_points:
    region = region_grow_single(image, seed, threshold=30)
    segmented_total = cv2.bitwise_or(segmented_total, region)

# Convert grayscale to RGB for red dots
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
for seed in seed_points:
    cv2.circle(image_rgb, (seed[1], seed[0]), radius=1, color=(255, 0, 0), thickness=-1)

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image with Seeds')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Region (2 Objects)')
plt.imshow(segmented_total, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("region_growing_segmentation_output.jpg")
print("Image saved as region_growing_segmentation_output.jpg")
plt.show()
