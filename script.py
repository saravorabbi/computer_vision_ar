import numpy as np
import cv2
from matplotlib import pyplot as plt

# reference frame - model image
reference = cv2.imread('./Data/ReferenceFrame.png')

# target image - image under analysis (target video)
target = cv2.VideoCapture('./Data/Multiple_View.avi')
if (target.isOpened() == False):
    print("Error opening video stream or file")

# augmented layer
layer = cv2.imread('./Data/AugmentedLayer.PNG')

# create detector
sift = cv2.xfeatures2d.SIFT_create()

# detect keypoints inside the reference image
kp_reference = sift.detect(reference)   # detection: individuo i keypoint
kp_reference, des_reference = sift.compute(reference, kp_reference)
# description: descrivo i kp secondo certe caratteristiche

while True:
    # capture frame-by-frame
    success, target_frame = target.read()

    # if frame is read correctly success is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    kp_target = sift.detect(target_frame)
    kp_target, des_target = sift.compute(target_frame, kp_target)

    # FLANN BASED MATCHER: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # create matcher
    matches = flann.knnMatch(des_reference, des_target, k=2)    # k: Finds the k best matches for each descriptor from a query set.
    good = []

    # Need to draw only good matches, so create a mask
    # inizializzo a zero una lista di lunghezza/punti uguale a matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test defined by Lowe in his SIFT paper
    for i, (m, n) in enumerate(matches):    # m: kp reference, n: kp target
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    draw_params = dict(matchColor=(255, 0, 255),
                       singlePointColor=(0, 255, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    kp_matches = cv2.drawMatchesKnn(target_frame, kp_target, reference, kp_reference, matches, None, **draw_params)

    cv2.imshow('kp correspondences', kp_matches)
    if cv2.waitKey(1) == ord('q'):
        break

    # Checking if we found the object
    MIN_MATCH_COUNT = 4
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_reference[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # print("src_pts: ", src_pts)
        # print("dst_pts: ", dst_pts)

        '''
        src_pts: prende da kp target tutti i punti che sono in good
        dst_pts: stessa cosa
        
        per la reshape:
        https://stackoverflow.com/questions/47402445/need-help-in-understanding-error-for-cv2-undistortpoints/47403282#47403282
        The short answer is you need to convert your points to a two-channel array of 32-bit floats as the error states:
        CV_32FC2, i.e. a three-dimensional array with shape (n_points, 1, 2).
        '''

        # Getting the coordinates of the corners of our query object in the train image
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # maschera il cambio di prospettiva
        h, w, _ = reference.shape
        # terza dimensione (_) sono i canali
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # pts: prendo i corners del target
        dst = cv2.perspectiveTransform(pts, H)
        # dst: intercetto i corner nel reference tramite H e trovo i pt
        # trasformazione di prospettiva tra target e reference

        # print("pts: ", pts)
        # print("dst: ", dst)

        print("Matches mask: ", matchesMask)

        h_t, w_t, _ = target_frame.shape
        h, w, _ = layer.shape

        # Getting the homography to project img_ar on the surface of the query object.
        pts_layer = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        H = cv2.getPerspectiveTransform(pts_layer, dst)

        # Warping the img_ar
        # Cambiamo effettivamente la prospettiva
        warped = cv2.warpPerspective(layer, H, (w_t, h_t))

        # Warp a white mask to understand what are the black pixels
        white = np.ones([h, w], dtype=np.uint8) * 255
        warp_mask = cv2.warpPerspective(white, H, (w_t, h_t))

        # Restore previous values of the train images where the mask is black
        warp_mask = np.equal(warp_mask, 0)
        warped[warp_mask] = target_frame[warp_mask]

        # Displaying the result
        cv2.imshow('Frame', warped)
        if cv2.waitKey(1) == ord('q'):
            break
    
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        break

# When everything done, release the capture
target.release()
cv2.destroyAllWindows()

'''
img_train -> target_frame (scena - video)
img_query -> reference (bishop - libro)
img_ar -> layer (stregatto - augmented layer)
'''
