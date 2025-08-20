# Fourier Transform Practice with Custom Image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image (use your own image, e.g., Okarun.jpeg)
img = cv.imread("Rudo.jpeg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# --- Fourier Transform using Numpy ---
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)


# --- Prepare all images for a single figure ---
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Numpy FT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# High Pass Filtering
fshift_hp = fshift.copy()
fshift_hp[crow - 30 : crow + 31, ccol - 30 : ccol + 31] = 0
f_ishift = np.fft.ifftshift(fshift_hp)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# OpenCV FT
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude, phase = cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum_cv = 20 * np.log(magnitude + 1)
phase_spectrum_cv = phase

# Low Pass Filtering (OpenCV)
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
fshift_lp = dft_shift * mask
f_ishift_lp = np.fft.ifftshift(fshift_lp)
img_back_lp = cv.idft(f_ishift_lp)
img_back_lp = cv.magnitude(img_back_lp[:, :, 0], img_back_lp[:, :, 1])

# --- Plot all results in one figure ---
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# Row 1: Numpy FT and HPF
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Input Image")
axes[0, 1].imshow(magnitude_spectrum, cmap="gray")
axes[0, 1].set_title("Magnitude Spectrum (Numpy)")
axes[0, 2].imshow(img_back, cmap="gray")
axes[0, 2].set_title("Image after HPF")
axes[0, 3].imshow(img_back, cmap="jet")
axes[0, 3].set_title("HPF Result (JET)")

# Row 2: OpenCV FT and LPF
axes[1, 0].imshow(magnitude_spectrum_cv, cmap="gray")
axes[1, 0].set_title("Magnitude Spectrum (OpenCV)")
axes[1, 1].imshow(phase_spectrum_cv, cmap="gray")
axes[1, 1].set_title("Phase Spectrum (OpenCV)")
axes[1, 2].imshow(img, cmap="gray")
axes[1, 2].set_title("Input Image (LPF)")
axes[1, 3].imshow(img_back_lp, cmap="gray")
axes[1, 3].set_title("Low Pass Filtered Image")

cv.imwrite("High_Pass_Filtered_Image.png", img_back)

for row in axes:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
