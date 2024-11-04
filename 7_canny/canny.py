import cv2 as cv
import sys
import random
import numpy as np


WNAME1 = "Image"
WNAME2 = "Canny"
WNAME3 = "Contours"
WNAME4 = "Canny-contours"

# 100, 200
minval = int(sys.argv[2])
maxval = int(sys.argv[3])

#img = cv.imread('./Nova.png')
img = cv.imread(sys.argv[1])

cv.namedWindow(WNAME1)
cv.imshow(WNAME1, img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img, minval, maxval)

edgesc = edges.copy()
contours, hierarchy = cv.findContours(edgesc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

imc = img.copy()
cv.drawContours(imc, contours, -1, (255, 0, 0), 1) 

cv.namedWindow(WNAME2)
cv.namedWindow(WNAME3)
cv.namedWindow(WNAME4)

cv.imshow(WNAME2, edges)
cv.imshow(WNAME3, imc)
cv.imshow(WNAME4, edgesc)

cv.waitKey(0)

cv.destroyAllWindows()
