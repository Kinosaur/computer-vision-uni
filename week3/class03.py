import cv2 as cv
import numpy as np

# Read the image
img = cv.imread("Okarun.jpeg")
assert img is not None, "file could not be read, check with os.path.exists()"

# 1. Custom 2D Convolution (Averaging Filter) - Stronger effect with larger kernel
kernel = np.ones((15, 15), np.float32) / 225
dst = cv.filter2D(img, -1, kernel)

# 2. Averaging (Normalized Box Filter) - Larger kernel
blur_avg = cv.blur(img, ksize=(111, 111))

# 3. Gaussian Blurring - Larger kernel
gaus_img = cv.GaussianBlur(img, ksize=(111, 111), sigmaX=0)

# 4. Median Blurring - Larger kernel
blur_median = cv.medianBlur(img, 15)

# 5. Bilateral Filtering - Stronger smoothing
blur_bilateral = cv.bilateralFilter(img, 25, 150, 150)

# 6. SHARPEN
sharpen_filter = np.array([[-1, -1, -1], [-1, 5, 0], [0, -1, 0]])
sharp_img = cv.filter2D(img, ddepth=-1, kernel=sharpen_filter)

# Display the results
cv.imshow("Original Image", img)
cv.imshow("Custom 2D Convolution", dst)
cv.imshow("Averaging Filter", blur_avg)
cv.imshow("Gaussian Blurring", gaus_img)
cv.imshow("Median Blurring", blur_median)
cv.imshow("Bilateral Filtering", blur_bilateral)
cv.imshow("Sharpened Image", sharp_img)

cv.waitKey(0)
cv.destroyAllWindows()

# Save the results
cv.imwrite("custom_2d_convolution.jpg", dst)
cv.imwrite("averaging_filter.jpg", blur_avg)
cv.imwrite("gaussian_blurring.jpg", gaus_img)
cv.imwrite("median_blurring.jpg", blur_median)
cv.imwrite("bilateral_filtering.jpg", blur_bilateral)
cv.imwrite("sharpened_image.jpg", sharp_img)
