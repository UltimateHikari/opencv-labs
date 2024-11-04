import numpy as np
import cv2 as cv
import sys

WNAME = "FLANN matching"

#img = cv.imread('./query.jpg')
img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
#img = cv.imread('./train.jpg')
train = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

# 5 50 
ftrees = int(sys.argv[3])
fchecks = int(sys.argv[4])
# 5 for visibility, 7 as in paper (it's a mess)
ratio = 0.1*int(sys.argv[5])

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(train,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = ftrees)
search_params = dict(checks=fchecks)

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < ratio*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

imgm = cv.drawMatchesKnn(img,kp1,train,kp2,matches,None,**draw_params)

cv.namedWindow(WNAME)
cv.imshow(WNAME, imgm)

cv.waitKey(0) 
cv.destroyAllWindows()

