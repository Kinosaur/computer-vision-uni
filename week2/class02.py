import cv2 as cv
import numpy as np

# Load the image
img_path = "Balkan.jpg"

img = cv.imread(img_path)

# Grayscale conversion
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blank = np.zeros(img.shape[:2], dtype="uint8")

# Color channel splitting
b, g, r = cv.split(img)
b_merge = cv.merge([b, blank, blank])

cv.imshow("Original Image", img)
cv.imshow("Grayscale Image", gray)
cv.imshow("Blue Channel", b_merge)

cv.waitKey(0)
cv.destroyAllWindows()
