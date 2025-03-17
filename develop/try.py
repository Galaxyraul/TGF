import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "./dataset_no_bc/drones/152.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to segment the object
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image for the shadow effect
shadow_image = np.zeros_like(image)

# Draw the contours to create the shadow
cv2.drawContours(shadow_image, contours, -1, (50, 50, 50), thickness=cv2.FILLED)

# Display the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(cv2.cvtColor(shadow_image, cv2.COLOR_BGR2RGB))
ax[1].set_title("Shadow Effect")
ax[1].axis("off")

plt.show()
