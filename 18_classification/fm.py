import numpy as np
import cv2 as cv
import sys
import imutils

WNAME1 = "badger matching"
WNAME2 = "chipmunk matching"

#img = cv.imread('./query.jpg')
img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
#img = cv.imread('./train.jpg')

# 200
threshold = int(sys.argv[2])
# 5 50 
ftrees = int(sys.argv[3])
fchecks = int(sys.argv[4])
# 5 for visibility, 7 as in paper (it's a mess)
ratio = 0.1*int(sys.argv[5])
# minwidth
mwidth = int(sys.argv[6])
# are samples blacked (black/no)
blacked = sys.argv[7]
# does query's 20% ground blacked (black/no)
grblacked = sys.argv[8]

badger_name = "./badger.png"
chip_name = "./burunduk.png"

if (blacked == "blacked"):
	badger_name = "./badger_b.png"
	chip_name = "./burunduk_b.png"


badg = cv.imread(badger_name, cv.IMREAD_GRAYSCALE)
chip = cv.imread(chip_name, cv.IMREAD_GRAYSCALE)

img = imutils.resize(img, width=mwidth)
badg = imutils.resize(badg, width=mwidth)
chip = imutils.resize(chip, width=mwidth)

h, w = img.shape[:2]
hb, _ = badg.shape[:2]
hc, _ = chip.shape[:2]
borderType = cv.BORDER_CONSTANT
img = cv.copyMakeBorder(img, int(hb-h), 0, 0, 0, borderType, (0,0,0))

print(h, hb, hc)
h, w = img.shape[:2]

if (grblacked == "blacked"):
	img = cv.rectangle(img, (int(0), int(0.9*h)), (int(w), int(h)), (0,0,0), -1)

# Initiate SIFT detector
sift = cv.SIFT_create(200)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(badg,None)
kp3, des3 = sift.detectAndCompute(chip,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = ftrees)
search_params = dict(checks=fchecks)

flann = cv.FlannBasedMatcher(index_params,search_params)

# badger test
matches_badg = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask_b = [[0,0] for i in range(len(matches_badg))]
matches_badger = 0

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches_badg):
	if m.distance < ratio*n.distance:
		matchesMask_b[i]=[1,0]
		matches_badger += 1
		print(m)

draw_params_b = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask_b,
                   flags = cv.DrawMatchesFlags_DEFAULT)

imgb = cv.drawMatchesKnn(img,kp1,badg,kp2,matches_badg,None,**draw_params_b)

cv.namedWindow(WNAME1)
cv.imshow(WNAME1, imgb)

# another resize

h, w = img.shape[:2]
hb, _ = badg.shape[:2]
hc, _ = chip.shape[:2]
borderType = cv.BORDER_CONSTANT
img = cv.copyMakeBorder(img, int(hc-h), 0, 0, 0, borderType, (0,0,0))

kp1, des1 = sift.detectAndCompute(img,None)

# chip test
matches_chip = flann.knnMatch(des1,des3,k=2)

# Need to draw only good matches, so create a mask
matchesMask_c = [[0,0] for i in range(len(matches_chip))]
matches_chipmunk = 0

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches_chip):
	if m.distance < ratio*n.distance:
		matchesMask_c[i]=[1,0]
		matches_chipmunk += 1

draw_params_c = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask_c,
                   flags = cv.DrawMatchesFlags_DEFAULT)

imgc = cv.drawMatchesKnn(img,kp1,chip,kp3,matches_chip,None,**draw_params_c)

cv.namedWindow(WNAME2)
cv.imshow(WNAME2, imgc)

print("Scores: badger: ", str(matches_badger), " chipmunk: ", str(matches_chipmunk), " Winner: ", ("badger" if matches_badger > matches_chipmunk else "chipmunk"))

cv.waitKey(0) 
cv.destroyAllWindows()

