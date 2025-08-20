import numpy as np
import cv2 as cv

img = cv.imread("Balkan.jpg")
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols = img.shape[:2]
height, width = img.shape[:2]

# Scaling
res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
cv.imshow("Original Image", img)
cv.imshow("Resized Image", res)

# cv.imwrite("Balkan_resized.jpg", res)

# Translation
M = np.float32([[1, 0, 100], [0, 1, 130]])
translated = cv.warpAffine(img, M, (cols, rows))

cv.imshow("Translated Image", translated)

# cv.imwrite("Balkan_translated.jpg", translated)

# Rotation
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
rotated = cv.warpAffine(img, M, (cols, rows))

cv.imshow("Rotated Image", rotated)

# cv.imwrite("Balkan_rotated.jpg", rotated)

# Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 200], [100, 250]])

M = cv.getAffineTransform(pts1, pts2)

affine = cv.warpAffine(img, M, (cols, rows))

cv.imshow("Affine Image", affine)

# cv.imwrite("Balkan_affine.jpg", affine)

# Perspective Transformation
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [600, 600]])

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(img, M, (300, 300))

cv.imshow("Perspective Image", dst)

# cv.imwrite("Balkan_perspective.jpg", dst)


cv.waitKey(0)
cv.destroyAllWindows()
