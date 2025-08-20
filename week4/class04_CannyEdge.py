import cv2 as cv
from matplotlib import pyplot as plt

# Read images from the 'pictures' folder
img1 = cv.imread("pictures/finalNoLight.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("pictures/fullLight.jpg", cv.IMREAD_GRAYSCALE)
img3 = cv.imread("pictures/noLight.jpg", cv.IMREAD_GRAYSCALE)
img4 = cv.imread("pictures/partial1Light.jpg", cv.IMREAD_GRAYSCALE)
img5 = cv.imread("pictures/partial2Light.jpg", cv.IMREAD_GRAYSCALE)
img6 = cv.imread("pictures/partialLight.jpg", cv.IMREAD_GRAYSCALE)

image_list = [img1, img2, img3, img4, img5, img6]
image_names = [
    "finalNoLight.jpg",
    "fullLight.jpg",
    "noLight.jpg",
    "partial1Light.jpg",
    "partial2Light.jpg",
    "partialLight.jpg",
]

plt.figure(figsize=(10, 2 * len(image_list)))
for i, (im, name) in enumerate(zip(image_list, image_names)):
    if im is not None:
        edges = cv.Canny(im, 100, 200)
        plt.subplot(len(image_list), 2, 2 * i + 1)
        plt.imshow(im, cmap="gray")
        plt.title(f"Original: {name}")
        plt.axis("off")
        plt.subplot(len(image_list), 2, 2 * i + 2)
        plt.imshow(edges, cmap="gray")
        plt.title(f"Edge: {name}")
        plt.axis("off")
plt.tight_layout()
plt.show()
