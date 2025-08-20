import cv2
from cv2 import dnn_superres
import matplotlib.pyplot as plt

# 1. Initialize the Super-Resolution engine
# This creates an object to handle the super-resolution tasks.
super_res = dnn_superres.DnnSuperResImpl_create()

# 2. Load the pre-trained model
# Define the path to your downloaded EDSR model file.
model_path = "superResModels/EDSR_x4.pb"

# Use readModel() to load the weights and architecture from the .pb file.
super_res.readModel(model_path)

# 3. Set the Model and Scale
# You must tell OpenCV which model architecture and scale you're using.
# For "EDSR_x4.pb", the model name is "edsr" and the scale is 4.
super_res.setModel("edsr", 4)


def downscaling(img_path):
    rows, cols, _ = map(int, img_path.shape)
    return cv2.resize(img_path, ((cols // 5), (rows // 5)))


# 4. Load your low-resolution image
# Load the image you want to enhance.
img = cv2.imread("tattoo/94aa687dbe2740a7363c603e13112462.jpg")
low_res_img = downscaling(img)

# 5. Upscale the image
# The upsample() method applies the loaded model to your image.
# This is where the magic happens and can take a moment to process.
high_res_img = super_res.upsample(low_res_img)

# 6. Display the results (optional)
# OpenCV loads images in BGR format, but matplotlib displays in RGB.
# We need to convert the color channels for correct display.
low_res_rgb = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2RGB)
high_res_rgb = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)

# Create a plot to compare the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"Low-Res: {low_res_rgb.shape[:2]}")
plt.imshow(low_res_rgb)

plt.subplot(1, 2, 2)
plt.title(f"EDSR x4 High-Res: {high_res_rgb.shape[:2]}")
plt.imshow(high_res_rgb)

plt.show()

# You can also save the final image
# Note: cv2.imwrite expects BGR format, which high_res_img is already in.
# cv2.imwrite("enhanced_image.png", high_res_img)

cv2.destroyAllWindows()  # Clean up any OpenCV windows if they were opened
cv2.waitKey(0)  # Wait for a key press to close the windows
