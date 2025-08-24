import cv2

# Read image
tattoo_path = "tattoo/Gao-Yord-Tattoo-Thailand Medium.jpeg"
img = cv2.imread(tattoo_path)


# Center crop and downscale
def center_crop_and_downscale(img, crop_h=100, crop_w=100, scale=1):
    h, w, _ = img.shape
    start_y = max((h - crop_h) // 2, 0)
    start_x = max((w - crop_w) // 2, 0)
    cropped = img[start_y : start_y + crop_h, start_x : start_x + crop_w]
    new_h, new_w = cropped.shape[0] // scale, cropped.shape[1] // scale
    return cv2.resize(cropped, (new_w, new_h))


low_res = center_crop_and_downscale(img)

# Super-resolution models
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# EDSR x4
sr.readModel("superResModels/EDSR_x4.pb")
sr.setModel("edsr", 4)
result_edsr = sr.upsample(low_res)
resized_edsr = cv2.resize(low_res, dsize=None, fx=4, fy=4)

# ESPCN x3
sr.readModel("superResModels/ESPCN_x3.pb")
sr.setModel("espcn", 3)
result_espcn = sr.upsample(low_res)
resized_espcn = cv2.resize(low_res, dsize=None, fx=3, fy=3)

# FSRCNN x3
sr.readModel("superResModels/FSRCNN_x3.pb")
sr.setModel("fsrcnn", 3)
result_fsrcnn = sr.upsample(low_res)
resized_fsrcnn = cv2.resize(low_res, dsize=None, fx=3, fy=3)

# LapSRN x8
sr.readModel("superResModels/LapSRN_x8.pb")
sr.setModel("lapsrn", 8)
result_lapsrn = sr.upsample(low_res)
resized_lapsrn = cv2.resize(low_res, dsize=None, fx=8, fy=8)

# Optionally, save results if needed
# cv2.imwrite("results/edsr_upscaled.png", result_edsr)
# cv2.imwrite("results/espcn_upscaled.png", result_espcn)
# cv2.imwrite("results/fsrcnn_upscaled.png", result_fsrcnn)
# cv2.imwrite("results/lapsrn_upscaled.png", result_lapsrn)
