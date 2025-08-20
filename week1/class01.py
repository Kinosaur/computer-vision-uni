import cv2

# Read the image
img = cv2.imread("./week1/catCool.jpg")

# Display the image in a window
cv2.imshow("cat", img)

# Wait for the user to close the window
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
