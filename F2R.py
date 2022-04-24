# Frame To Reference
import numpy as np
import cv2

# reference frame - model image -> foto del libro
reference = cv2.imread('./Data/ReferenceFrame.png')
 
#portiamo la maschera da BGR a scala di grigio -> da 3 canali a un canale
reference_alpha_mask = cv2.cvtColor(cv2.imread('./Data/ObjectMask.PNG'), cv2.COLOR_BGR2GRAY)

# splitto la reference in 3 canali
b_r, g_r, r_r = cv2.split(reference)

# creo lista in cui aggiungo il quarto canale alpha ai 3 canali appena splittati
# alpha 255-> opacità massima
#       0 -> opacità minima (diventa trasparente)
rgba_r = [b_r, g_r, r_r, reference_alpha_mask]

# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga7d7b4d6c6ee504b30a20b1680029c7b4  merge

# faccio il merge dei 4 canali e li salvo la nuova img "ReferenceFrameAlpha"
cv2.imwrite('./Data/ReferenceFrameAlpha.png', cv2.merge(rgba_r, 4))
reference_rgba = cv2.imread('./Data/ReferenceFrameAlpha.png')


# target image - image under analysis -> (target video)
target = cv2.VideoCapture('./Data/Multiple_View.avi')
if (target.isOpened() == False):
    print("Error opening video stream or file")


# augmented layer -> layer da soraimporre coi loghi
layer = cv2.imread('./Data/AugmentedLayer.PNG')
# conversione da 3 canali a 1 canale
layer_alpha_mask = cv2.cvtColor(cv2.imread('./Data/AugmentedLayerMask.PNG'), cv2.COLOR_BGR2GRAY)
# split del layer in 3 canali
b_l, g_l, r_l = cv2.split(layer)
# aggiungo canale alpha
rgba_l = [b_l, g_l, r_l, layer_alpha_mask]
# per come è fatta la imwrite salva in 3 canali, ma mantiene il canale alpha
cv2.imwrite('./Data/AugmentedLayerAlpha.PNG', cv2.merge(rgba_l, 4))
layer_rgba = cv2.imread('./Data/AugmentedLayerAlpha.PNG')


# resize the layer so that it has the same shape of the reference image
h_r, w_r, _ = reference_rgba.shape 
layer_rgba_resized = layer_rgba[0:0 + h_r, 0:0 + w_r]


# create detector -> trova i keypoints
sift = cv2.SIFT_create()

# detect keypoints inside the reference image
kp_reference = sift.detect(reference_rgba)  # detection: individuo i keypoint
kp_reference, des_reference = sift.compute(reference_rgba, kp_reference)
# des_reference = descriptor: descrive i kp secondo certe caratteristiche
# kp_reference = keypoints effettivi, sono coordinate dell'immagine

video_ar = []

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
    matches = flann.knnMatch(des_reference, des_target, k=2)  # k: Finds the k best matches for each descriptor from a query set.
    
    # lista in cui vengono inseriti i keypoints matchati
    good = []

    # ratio test defined by Lowe in his SIFT paper
    # rapporto tra kp ref e kp target < 0.7 -> permette d i defnire se le corrispondenze sono buone
    for i, (m, n) in enumerate(matches):  # m: kp reference, n: kp target
        if m.distance < 0.7 * n.distance:
            good.append(m) # se si -> metto aggiungo alla lista

# codice vero da qui

    # Checking if we found the object
    MIN_MATCH_COUNT = 4 # numero minimo di corrispondenze buone che accettiamo
    if len(good) > MIN_MATCH_COUNT:
        # trasformo in numeri reali i kp della reference e del target frame
        '''
        src_pts: prende da kp target tutti i punti che sono in good
        dst_pts: stessa cosa

        per la reshape:
        https://stackoverflow.com/questions/47402445/need-help-in-understanding-error-for-cv2-undistortpoints/47403282#47403282
        The short answer is you need to convert your points to a two-channel array of 32-bit floats as the error states:
        CV_32FC2, i.e. a three-dimensional array with shape (n_points, 1, 2).
        '''
        src_pts = np.float32([kp_reference[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        

        # Getting the coordinates of the corners of our query object in the train image
        # trovo la funzione di omografia -> fra pt sorgente e pt di destinazione
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist() # si puo' anche togliere
        
        # maschera il cambio di prospettiva
        # corners (4) della reference -> li storizzo con il formato giusto
        pts = np.float32([[0, 0], [0, h_r - 1], [w_r - 1, h_r - 1], [w_r - 1, 0]]).reshape(-1, 1, 2)
        
        # dst: intercetto i corner nel target tramite H e trovo i pt
        # trasformazione di prospettiva tra target e reference
        # prendo i corner della reference e tramite la H trovo la corrispondenca dei corner nel video (target frame)
        dst = cv2.perspectiveTransform(pts, H) #corner sul frame

        h_t, w_t, _ = target_frame.shape
        h_l, w_l, _ = layer_rgba_resized.shape

        # Getting the homography to project img_ar on the surface of the query object.
        # prendo i corner  del layer
        pts_layer = np.float32([[0, 0], [0, h_l - 1], [w_l - 1, h_l - 1], [w_l - 1, 0]]).reshape(-1, 1, 2)
        H = cv2.getPerspectiveTransform(pts_layer, dst)

        # Warping the img_ar
        # Cambiamo effettivamente la prospettiva
        warped = cv2.warpPerspective(layer_rgba_resized, H, (w_t, h_t))

        # Restore previous values of the train images where the mask is black
        # in warp_mask ci sono tutti i pt in cui la maschera e' trasparente (dove e' uguale a =0)
        warp_mask = np.equal(warped, 0)
        # modifichiamo warped (che e' il layer trasformato di prospettiva) e ci piazziamo dentro i pixel del target frame (dato che altrimenti stamperemmo nero)
        warped[warp_mask] = target_frame[warp_mask]

        video_ar.append(warped)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        break


out = cv2.VideoWriter('Augmented_Multiple_View_F2R.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w_t, h_t))
 
for i in range(len(video_ar)):
    out.write(video_ar[i])
    ####
    status=i/419*100
    print("Building video:", round(status, 2), "%\n")
    ####
out.release()