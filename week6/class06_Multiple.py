import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read the main image and template
img_rgb = cv.imread("mario_coins.png")
assert img_rgb is not None, "Main image could not be read."
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread(
    "template2.png", cv.IMREAD_GRAYSCALE
)  # Replace with your actual filename if needed
assert template is not None, "Template image could not be read."
w, h = template.shape[::-1]

# Perform template matching
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

# Show result
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title("Detected Coins")
plt.axis("off")
plt.show()

cv.imwrite("detected_coins.png", img_rgb)  # Save the result image
