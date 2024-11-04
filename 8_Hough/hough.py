import cv2 as cv
import sys
import random
import numpy as np


WNAME1 = "HoughLinesP"

# 20, 200
minval = int(sys.argv[2])
maxval = int(sys.argv[3])
# 60, 30, 3
thresh = int(sys.argv[4])
minlen = int(sys.argv[5])
maxgap = int(sys.argv[6])

lc = (0,0,255)
lsize = 2

#img = cv.imread('./Nova.png')
img = cv.imread(sys.argv[1])

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img, minval, maxval)

cedg = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
cedgb = cedg.copy()

linesP = cv.HoughLinesP(edges, 1, np.pi / 180, thresh, None, minlen, maxgap)

img_res = img.copy()

if linesP is not None:
	for i in range(0, len(linesP)):
		l = linesP[i][0]
		cv.line(cedg, (l[0], l[1]), (l[2], l[3]), lc, lsize, cv.LINE_AA)
		cv.line(img_res, (l[0], l[1]), (l[2], l[3]), lc, lsize, cv.LINE_AA)

cv.namedWindow(WNAME1)
cv.imshow(WNAME1,cv.hconcat([img, cedgb, cedg, img_res]))

cv.waitKey(0)

cv.destroyAllWindows()
