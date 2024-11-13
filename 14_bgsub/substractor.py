import numpy as np 
import cv2 as cv
import sys

cap = cv.VideoCapture(sys.argv[1]) 

# 20 for forgetting letters, 500 for default
hist = int(sys.argv[2])
# initializing subtractor  
fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=hist)
fgbg2 = cv.createBackgroundSubtractorMOG2(history=hist)
fgbgknn = cv.createBackgroundSubtractorKNN(history=hist)
fgbggmg = cv.bgsegm.createBackgroundSubtractorGMG()

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) 

WNAME1 = "MOG"
WNAME2 = "MOG2"
WNAMEG = "GMG"
WNAMEK = "KNN"

cv.namedWindow(WNAME1)
cv.namedWindow(WNAME2)
cv.namedWindow(WNAMEG)
cv.namedWindow(WNAMEK)
  
while(1): 
	ret, frame = cap.read()        

	fgmask = fgbg.apply(frame)   
	cv.imshow(WNAME1, fgmask) 

	fgmask2 = fgbg2.apply(frame)   
	cv.imshow(WNAME2, fgmask2) 
	
	fgmaskgmg = fgbggmg.apply(frame)   
	fgmaskgmg = cv.morphologyEx(fgmaskgmg, cv.MORPH_OPEN, kernel)
	cv.imshow(WNAMEG, fgmaskgmg) 
	
	fgmaskk = fgbgknn.apply(frame)   
	cv.imshow(WNAMEK, fgmaskk) 

	k = cv.waitKey(30) & 0xff
	if k == 27: 
		break
  
cap.release() 
cv.destroyAllWindows() 

