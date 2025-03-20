import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load images
original_img_path = "watermark-maker.jpg"  # Original image with watermark
modified_img_path = "wmremove-transformed.jpg"  # Watermark-removed image

original = cv2.imread(original_img_path)
modified = cv2.imread(modified_img_path)

# Convert to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)

# Compute SSIM (Structural Similarity Index)
score, diff = ssim(original_gray, modified_gray, full=True)
diff = (diff * 255).astype("uint8")  # Normalize the difference image

# Threshold the difference to highlight watermark removal regions
_, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours of the changed regions
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the modified image to highlight removed areas
output = modified.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Ignore small noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display results
cv2.imshow("Original", original)
cv2.imshow("Modified", modified)
cv2.imshow("Difference", diff)
cv2.imshow("Detected Changes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
