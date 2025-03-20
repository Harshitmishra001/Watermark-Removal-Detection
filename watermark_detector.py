import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== Load and Preprocess Image ==========
def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return image, gray

# ========== Edge Detection ==========
def detect_edges(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges
    return edges

# ========== Texture Analysis (Laplacian Variance) ==========
def texture_analysis(gray_image):
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()  # Variance of Laplacian
    return variance

# ========== Frequency Analysis (DFT) ==========
def apply_dft(gray_image):
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # Shift to center
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) 
    return magnitude_spectrum

# ========== Detect Watermark Removal ==========
def detect_watermark_removal(gray_image):
    texture_var = texture_analysis(gray_image)  # Get texture variation

    # Define texture threshold (adjust based on testing)
    watermark_removed = texture_var < 50.0  # Lower variance → Smoother (watermark likely removed)
    return watermark_removed

# ========== Main Execution ==========
image_path = "wmremove-transformed.jpg"  # Replace with actual image path

# Load image
image, gray = load_image(image_path)

# Edge detection
edges = detect_edges(gray)

# Frequency analysis
magnitude_spectrum = apply_dft(gray)

# Watermark removal detection
watermark_removed = detect_watermark_removal(gray)

# ========== Display Results ==========
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")

plt.subplot(1, 3, 3)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Frequency Spectrum (DFT)")

plt.tight_layout()
plt.show()

# Print Result
if watermark_removed:
    print("Warning: Watermark appears to be removed!")
else:
    print("Watermark is still present in the image.")
