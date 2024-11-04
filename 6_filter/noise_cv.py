import cv2 as cv
import sys
import random
import numpy as np


WNAME = "Noise filters"

# noise type: 0 - randu, 1 - randn
t = int(sys.argv[2])
# noise param 1 (mean)
m = int(sys.argv[3])
# noise param 2 (sigma or high)
s = int(sys.argv[4])
# denoise kernel size
k = int(sys.argv[5])

#img = cv.imread('./Lena512.png')
img = cv.imread(sys.argv[1])

#img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

mv = (m,m,m) 
sv = (s,s,s)

if t > 0:
	# add gaussian
	noise = cv.randn(img.copy(), mv, sv)
else:
	noise = cv.randu(img.copy(), mv, sv)	

print (img.shape, noise.shape)

img_res = cv.add(img, noise)

ksize = (k,k)
img_blur = cv.blur(img, ksize)

img_med = cv.medianBlur(img, k)

kernel = np.ones((k,k),np.float32)/(k*k)
img_filter = cv.filter2D(img,-1, kernel)

cv.namedWindow(WNAME)

cv.imshow(WNAME, cv.hconcat([cv.hconcat([img_res, img_blur]), cv.hconcat([img_res, img_med]),cv.hconcat([img_res, img_filter])]))

cv.waitKey(0)

cv.destroyAllWindows()
