# Frame To Reference
import numpy as np
import cv2

# load reference frame
reference = cv2.imread('./Data/ReferenceFrame.png')
# converting mask from 3 to 1 channel
reference_alpha_mask = cv2.cvtColor(cv2.imread('./Data/ObjectMask.PNG'), cv2.COLOR_BGR2GRAY)
# split reference in 3 channels
b_r, g_r, r_r = cv2.split(reference)
# add the alpha channel
rgba_r = [b_r, g_r, r_r, reference_alpha_mask]
# merge of the 4 channel
cv2.imwrite('./Data/ReferenceFrameAlpha.png', cv2.merge(rgba_r, 4))
reference_rgba = cv2.imread('./Data/ReferenceFrameAlpha.png')


# load target video
target = cv2.VideoCapture('./Data/Multiple_View.avi')
if (target.isOpened() == False):
    print("Error opening video stream")


# load augmented layer
layer = cv2.imread('./Data/AugmentedLayer.PNG')
# converting mask from 3 to 1 channel
layer_alpha_mask = cv2.cvtColor(cv2.imread('./Data/AugmentedLayerMask.PNG'), cv2.COLOR_BGR2GRAY)
# split layer in 3 channels
b_l, g_l, r_l = cv2.split(layer)
# add the alpha channel
rgba_l = [b_l, g_l, r_l, layer_alpha_mask]
# merge of the 4 channel
cv2.imwrite('./Data/AugmentedLayerAlpha.PNG', cv2.merge(rgba_l, 4))
layer_rgba = cv2.imread('./Data/AugmentedLayerAlpha.PNG')


# resizing the layer to the reference hight and width
h_ref, w_ref, _ = reference_rgba.shape 
layer_rgba_resized = layer_rgba[0:0 + h_ref, 0:0 + w_ref]


#sift = cv2.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create()

# detect keypoints inside the reference image
kp_reference = sift.detect(reference_rgba)
kp_reference, des_reference = sift.compute(reference_rgba, kp_reference)

video_ar = []

while True:
    # read frame-by-frame
    success, target_frame = target.read()

    # control if frame is correctly read
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
	
	# find current frame kp
    kp_target = sift.detect(target_frame)
    kp_target, des_target = sift.compute(target_frame, kp_target)

    # flann based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
	# finds the k best matches for each descriptor from a query set
    matches = flann.knnMatch(des_reference, des_target, k=2)

    # list of matched keypoints
    good = []

	# test good correspondence between kp_ref and kp_target as defined by Lowe in his SIFT paper
    for i, (m, n) in enumerate(matches):  # m: kp reference, n: kp target
        if m.distance < 0.7 * n.distance:
            good.append(m)

	# min number of good correspondence
    MIN_MATCH_COUNT = 4
    if len(good) > MIN_MATCH_COUNT:
        # cast keypoints to real number 
        src_pts = np.float32([kp_reference[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
		# find homography trough kp_reference and kp_target
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # reference corners as float
        pts = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
        # take the reference corner and trough H find the correponding corner in the target frame
        dst = cv2.perspectiveTransform(pts, H)

        h_frame, w_frame, _ = target_frame.shape
        h_layer, w_layer, _ = layer_rgba_resized.shape

        # layer corners as float
        pts_layer = np.float32([[0, 0], [0, h_layer - 1], [w_layer - 1, h_layer - 1], [w_layer - 1, 0]]).reshape(-1, 1, 2)
        # getting the homography to project the layer on the surface of the frame
        H = cv2.getPerspectiveTransform(pts_layer, dst)

        # warping the layer
        warped = cv2.warpPerspective(layer_rgba_resized, H, (w_frame, h_frame))

        # Restore previous values of the frame where the warped layer is black
        warp_mask = np.equal(warped, 0)
        warped[warp_mask] = target_frame[warp_mask]

        video_ar.append(warped)

    else:
        print("Not enough matches found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        break


out = cv2.VideoWriter('Augmented_Multiple_View_F2R.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w_frame, h_frame))

for i in range(len(video_ar)):
	print("i = ", i)

for i in range(len(video_ar)):
    out.write(video_ar[i])
    ####
    status=i/419*100
    print("Building video:", round(status, 2), "%\n")
    ####
out.release()

