# Frame To Frame
import numpy as np
import cv2

# load images
layer = cv2.imread('./Data/AugmentedLayer.PNG')
layer_mask = cv2.imread('./Data/AugmentedLayerMask.PNG', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread('./Data/ReferenceFrame.png')
reference_mask = cv2.imread('./Data/ObjectMask.PNG', cv2.IMREAD_GRAYSCALE)
video = cv2.VideoCapture('./Data/Multiple_View.avi')
if (video.isOpened() == False):
    print("Error opening video stream")

# layer preprocessing: alpha channel + resizing
b_layer, g_layer, r_layer = cv2.split(layer)
layer = [b_layer, g_layer, r_layer, layer_mask]
cv2.imwrite('./Data/AugmentedLayerAlpha.PNG', cv2.merge(layer, 4))
layer = cv2.imread('./Data/AugmentedLayerAlpha.PNG')

reference_height, reference_weight = reference.shape[:2]
layer = layer[0:0 + reference_height, 0:0 + reference_weight]
layer_mask = layer_mask[0:0 + reference_height, 0:0 + reference_weight]

# find reference kp
sift = cv2.SIFT_create()
kp_reference = sift.detect(reference, reference_mask)
kp_reference, des_reference = sift.compute(reference, kp_reference)

# find layer kp
kp_layer = sift.detect(layer, layer_mask)
kp_layer, des_layer = sift.compute(layer, kp_layer)

#find matches
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_layer, des_layer,k=2)

good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = np.float32([kp_layer[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp_reference[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H_layer, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# pts = np.float32([[0, 0], [0, reference_height - 1], [reference_weight - 1, reference_height - 1], [reference_weight - 1, 0]]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts, H_layer) #corner sul frame
# layer_height, layer_weight = layer.shape[:2]
# pts = np.float32([[0, 0], [0, layer_height - 1], [layer_weight - 1, layer_weight - 1], [layer_weight - 1, 0]]).reshape(-1, 1, 2)
# H_layer = cv2.getPerspectiveTransform(pts, dst)

video_ar = []

while True:
    success, current_frame = video.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # find kp current frame
    kp_current_frame = sift.detect(current_frame)
    kp_current_frame, des_current_frame = sift.compute(current_frame, kp_current_frame)
    
    # find matches between reference and current frame
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_reference, des_current_frame, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_reference[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_current_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # reference_height, reference_weight = reference.shape[:2]
        # pts = np.float32([[0, 0], [0, reference_height - 1], [reference_weight - 1, reference_height - 1], [reference_weight - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, H) 

        # pts = np.float32([[0, 0], [0, layer_height - 1], [layer_weight - 1, layer_height - 1], [layer_weight - 1, 0]]).reshape(-1, 1, 2)
        # pts_layer = np.float32(cv2.perspectiveTransform(pts, H_layer).reshape(-1, 1, 2))
        # H = cv2.getPerspectiveTransform(pts_layer, dst)

        M_mask=np.dot(np.identity(3, dtype='float64'), H)
        M = np.dot(H, H_layer)

        warped = cv2.warpPerspective(layer, M, (reference_weight, reference_height))

        warp_mask = cv2.warpPerspective(layer_mask, M, (reference_weight, reference_height))

        warp_mask = np.equal(warped, 0)
        warped[warp_mask] = current_frame[warp_mask]

        video_ar.append(warped)

        H_layer = H
        reference = current_frame

        img_maskTrans = cv2.warpPerspective(reference_mask, M_mask, (reference_weight, reference_height), flags= cv2.INTER_NEAREST)
        
        kp_reference=sift.detect(reference, img_maskTrans)
        kp_reference, des_reference = sift.compute(reference, kp_reference)

        #kp_reference = kp_current_frame
        #des_reference = des_current_frame

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