import cv2
import numpy as np

# Load the image in grayscale to calculate brightness
image = cv2.imread('833e5b65544e282ce5757cfc3eba73d.png', cv2.IMREAD_GRAYSCALE)

# Calculate the current average brightness
current_brightness = np.mean(image)

# Define your target brightness level (0-255 for 8-bit images)
target_brightness = 150  # for example

# Calculate the scaling factor
scaling_factor = target_brightness / current_brightness

# Scale the pixel values to reach the desired brightness level
# Clip the values to ensure they remain in the 8-bit range [0, 255]
rescaled_image = np.clip(image * scaling_factor, 0, 255).astype(np.uint8)

# If the original image is in color, we need to apply the same scaling to all channels
if len(image.shape) == 3:
    rescaled_image = cv2.merge([np.clip(channel * scaling_factor, 0, 255).astype(np.uint8)
                                for channel in cv2.split(image)])

# Save or display the adjusted image
cv2.imwrite('rescaled_image.jpg', rescaled_image)
# cv2.imshow('Rescaled Image', rescaled_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
