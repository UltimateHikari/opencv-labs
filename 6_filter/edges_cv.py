import cv2 as cv
import sys
import random
import numpy as np


WNAME1 = "Image"
WNAME2 = "Blurred"
WNAME3 = "Sobel filters"
WNAME4 = "Laplacian filters"

# scale = 1
s = int(sys.argv[2])
# delta = 0
d = int(sys.argv[3])
# kernelsize = 3
k = int(sys.argv[4])

ddepth = cv.CV_16S

#img = cv.imread('./Nova.png')
img = cv.imread(sys.argv[1])
cv.namedWindow(WNAME1)
cv.imshow(WNAME1, img)

img = cv.GaussianBlur(img, (3, 3), 0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=k, scale=s, delta=d, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=k, scale=s, delta=d, borderType=cv.BORDER_DEFAULT)

cx = cv.convertScaleAbs(grad_x)
cy = cv.convertScaleAbs(grad_y)
grad = cv.addWeighted(cx, 0.5, cy, 0.5, 0)

dst = cv.Laplacian(gray, ddepth, ksize=k)

cv.namedWindow(WNAME2)
cv.namedWindow(WNAME3)
cv.namedWindow(WNAME4)

cv.imshow(WNAME2, gray)
cv.imshow(WNAME3, cv.hconcat([cx, cy, grad]))
cv.imshow(WNAME4, cv.convertScaleAbs(dst))

cv.waitKey(0)

cv.destroyAllWindows()
