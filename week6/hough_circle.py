import numpy as np
import cv2 as cv

img = cv.imread("mario_circles.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Apply Gaussian blur instead of median blur for better circle detection
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Improved parameters for better circle detection
circles = cv.HoughCircles(
    img,
    cv.HOUGH_GRADIENT,
    dp=1,  # Inverse ratio of accumulator resolution
    minDist=50,  # Minimum distance between circle centers
    param1=50,  # Upper threshold for edge detection
    param2=50,  # Accumulator threshold for center detection
    minRadius=5,  # Minimum circle radius
    maxRadius=100,
)  # Maximum circle radius

# Check if any circles were detected
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f"Detected {len(circles[0])} circles")
else:
    print("No circles detected")

cv.imshow("detected circles", cimg)
cv.imwrite("detected_circles.png", cimg)  # Save the result image
cv.waitKey(0)
cv.destroyAllWindows()
