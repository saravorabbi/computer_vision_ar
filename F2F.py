# Frame To Frame
import numpy as np
import cv2

# load reference LIBRO
reference = cv2.imread('./Data/ReferenceFrame.png')
reference_mask = cv2.imread('./Data/ObjectMask.PNG', cv2.IMREAD_GRAYSCALE)      # rgb to grayscale
b, g, r = cv2.split(reference)
reference = [b, g, r, reference_mask]
cv2.imwrite('./Data/ReferenceFrameAlpha.png', cv2.merge(reference, 4))
reference = cv2.imread('./Data/ReferenceFrameAlpha.png')

# load video
video = cv2.VideoCapture('./Data/Multiple_View.avi')
if (video.isOpened() == False):
    print("Error opening video stream")

#load layer
layer = cv2.imread('./Data/AugmentedLayer.PNG')
layer_mask = cv2.imread('./Data/AugmentedLayerMask.PNG', cv2.IMREAD_GRAYSCALE)  # rgb to grayscale
b, g, r = cv2.split(layer)
layer = [b, g, r, layer_mask]
cv2.imwrite('./Data/AugmentedLayerAlpha.PNG', cv2.merge(layer, 4))
layer = cv2.imread('./Data/AugmentedLayerAlpha.PNG')
# reshape AR layer
reference_height, reference_weight = reference.shape[:2]
layer = layer[0:0 + reference_height, 0:0 + reference_weight]
layer_mask = layer_mask[0:0 + reference_height, 0:0 + reference_weight]

layer_height, layer_weight = layer.shape[:2]



### layer to reference frame homography ###

# find reference kp
#sift = cv2.SIFT_create() # python 3.8
sift = cv2.xfeatures2d.SIFT_create() # python 3.7

# detect keypoints inside the reference image
kp_reference = sift.detect(reference)
kp_reference, des_reference = sift.compute(reference, kp_reference)

video_ar = []

H_previous = np.identity(3, dtype='float64')     # identity matrix needed for first iteration homography

while True:
	# read frame-by-frame
    success, current_frame = video.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # find kp current frame
    kp_current_frame = sift.detect(current_frame)
    kp_current_frame, des_current_frame = sift.compute(current_frame, kp_current_frame)
    
    # flann based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
	# finds the k best matches for each descriptor from a query set
    matches = flann.knnMatch(des_reference, des_current_frame, k=2)

	# list of matched keypoints
    good = []

	# test good correspondence between kp_reference and kp_current_frame as defined by Lowe in his SIFT paper
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

	# min number of good correspondence
    MIN_MATCH_COUNT = 4
    if len(good) > MIN_MATCH_COUNT:
		# cast keypoints to real number
        src_pts = np.float32([kp_reference[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_current_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		# find homography trough kp_reference and kp_current_frame -> current transformation between this frame and the previous one
        H_current, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # accumulate the current transformation with the previous ones using dot product
        H_current = np.dot(H_previous, H_current)


		# reference corners as float
        pts = np.float32([[0, 0], [0, reference_height - 1], [reference_weight - 1, reference_height - 1], [reference_weight - 1, 0]]).reshape(-1, 1, 2)
        # take the reference corner and trough H find the correponding corner in the current frame
        dst = cv2.perspectiveTransform(pts, H_current) 
        
        # layer corners as float
        pts_layer = np.float32([[0, 0], [0, layer_height - 1], [layer_weight - 1, layer_height - 1], [layer_weight - 1, 0]]).reshape(-1, 1, 2)
        # getting the homography to project the layer on the surface of the frame
        H_layer = cv2.getPerspectiveTransform(pts_layer, dst)


		# warping the layer and the mask
        warped = cv2.warpPerspective(layer, H_layer, (reference_weight, reference_height))
        warp_mask = cv2.warpPerspective(layer_mask, H_layer, (reference_weight, reference_height))

		# restore previous values of the frame where the warped layer is black
        warp_mask = np.equal(warped, 0)
        warped[warp_mask] = current_frame[warp_mask]

        video_ar.append(warped)
		
		### pre processing current frame for next iteration
        # warping current frame mask 
        reference_mask = cv2.warpPerspective(reference_mask, H_current, (reference_weight, reference_height))
        # create frame with alpha channel using the reference_mask just found
        reference = current_frame
        b, g, r = cv2.split(reference)
        reference = [b, g, r, reference_mask]
        cv2.imwrite('./Data/ReferenceFrameAlpha.PNG', cv2.merge(reference, 4))
        reference = cv2.imread('./Data/ReferenceFrameAlpha.PNG')
        # find new kp
        kp_reference = sift.detect(reference)
        kp_reference, des_reference = sift.compute(reference, kp_reference)

        # assing current homography to the old one
        H_previous = H_current

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        break

out = cv2.VideoWriter('Augmented_Multiple_View_F2F.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (reference_weight, reference_height))
 
for i in range(len(video_ar)):
    out.write(video_ar[i])
    ####
    status=i/419*100
    print("Building video:", round(status, 2), "%\n")
    ####
out.release()