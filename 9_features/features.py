# source for warping: https://pavelk.ru/opencv-warpperspective-bez-obrezki-whole-image-i-razmer-rezultata-destination-result-size/
import cv2 as cv
import sys
import random
import numpy as np


WNAME1 = "CornersBasic"
WNAME2 = "Warped"
WNAME3 = "CornersWarped"
WNAME4 = "ShiBasic"
WNAME5 = "ShiWarped"

# 2 7 4 10 5
blockSize = int(sys.argv[2])
ksize = int(sys.argv[3])
k = 0.01 * int(sys.argv[4])
qualitylevel = 0.01 * int(sys.argv[5])
mindistance = int(sys.argv[6])

#img = cv.imread('./orokin720.png')
img = cv.imread(sys.argv[1])

# harris without warp
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners = cv.cornerHarris(gray, blockSize, ksize, k)
corners = cv.dilate(corners,None)
corn_res = img.copy()
corn_res[corners>0.05*corners.max()]=[0,0,255]

cv.namedWindow(WNAME1)
cv.imshow(WNAME1,cv.hconcat([img, corn_res]))

# warp for floor
imgh,imgw,imgc = img.shape
print (imgh, imgw)

pts1 = np.float32([[618, 276],[753, 294], [434, 435], [650, 492]])
pts2 = np.float32([[0, 0], [200, 0], [0, 300], [200,300]])
ptsc = np.float32(np.array([[[0, 0], [imgw, 0], [0, imgh], [imgw, imgh]]]))
M = cv.getPerspectiveTransform(pts1, pts2)

print (M)

ptsct = cv.perspectiveTransform(ptsc, M)
x,y,w,h = cv.boundingRect(ptsct)

print (x,y,w,h)

x = 1000
y = 1000
w = 1920
h = 1920

for i in pts2:
	i[0] += x
	i[1] += y

print(pts2)

M = cv.getPerspectiveTransform(pts1, pts2)
imgwarped  = cv.warpPerspective(img, M, (w,h))

cv.namedWindow(WNAME2)
cv.imshow(WNAME2, imgwarped)

# harris with warp
grayw = cv.cvtColor(imgwarped, cv.COLOR_BGR2GRAY)
cornersw = cv.cornerHarris(grayw, blockSize, ksize, k)
cornersw = cv.dilate(cornersw,None)
corn_resw = imgwarped.copy()
corn_resw[cornersw>0.05*cornersw.max()]=[0,0,255]

cv.namedWindow(WNAME3)
cv.imshow(WNAME3, corn_resw)

# shi without warp

grays = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cornerss = cv.goodFeaturesToTrack(grays,0,qualitylevel,mindistance)
cornerss = np.intp(cornerss)
imgs = img.copy()
for i in cornerss:
    x,y = i.ravel()
    cv.circle(imgs,(x,y),3,255,-1)

cv.namedWindow(WNAME4)
cv.imshow(WNAME4, imgs)

# shi with warp
graysw = cv.cvtColor(imgwarped,cv.COLOR_BGR2GRAY)
cornerssw = cv.goodFeaturesToTrack(graysw,0,qualitylevel,mindistance)
cornerssw = np.intp(cornerssw)
imgsw = imgwarped.copy()
for i in cornerssw:
    x,y = i.ravel()
    cv.circle(imgsw,(x,y),3,255,-1)

cv.namedWindow(WNAME5)
cv.imshow(WNAME5, imgsw)

cv.waitKey(0)

cv.destroyAllWindows()
