# source: https://docs.opencv.org/3.1.0/d9/dc8/tutorial_py_trackbar.html
import cv2
import numpy as np
import sys

def nothing(x):
	pass

TBMAX = 255
WNAME = 'Brightness, Contrast, Gamma'

def bsg(event=0): 
	b = cv2.getTrackbarPos('brightness', WNAME) - TBMAX
	s = cv2.getTrackbarPos('contrast', WNAME) - TBMAX
	g = cv2.getTrackbarPos('gamma', WNAME) - TBMAX
	effect = controller(img, b, c, g) 
	cv2.imshow('Effect', effect) 
  
def controller(img, brightness=255, 
               contrast=127): 
    
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
  
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 
  
    if brightness != 0: 
  
        if brightness > 0: 
  
            shadow = brightness 
  
            max = 255
  
        else: 
  
            shadow = 0
            max = 255 + brightness 
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow 
  
        cal = cv2.addWeighted(img, al_pha,  
                              img, 0, ga_mma) 
  
    else: 
        cal = img 
  
	if contrast != 0: 
		Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
		Gamma = 127 * (1 - Alpha) 

		cal = cv2.addWeighted(cal, Alpha,  
                              cal, 0, Gamma) 
  
    return cal 
  
if __name__ == '__main__': 
	img = cv2.imread(sys.argv[1])
	
	cv2.namedWindow(WNAME) 
	
	cv2.createTrackbar('brightness', WNAME, TBMAX*2, TBMAX*4, bsg)
	cv2.createTrackbar('contrast', WNAME, TBMAX, TBMAX*2, bsg)
	cv2.createTrackbar('gamma', WNAME, 0, TBMAX*2, nothing)
     
 	bsg(0)

cv2.waitKey(0) 













cv2.createTrackbar('brightness', WNAME, 0, TBMAX*2, nothing)
cv2.createTrackbar('contrast', WNAME, 0, TBMAX*2, nothing)
cv2.createTrackbar('gamma', WNAME, 0, TBMAX*2, nothing)

cv2.imshow('image',img)


while(1):
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	b = cv2.getTrackbarPos('brightness', WNAME) - TBMAX
	s = cv2.getTrackbarPos('contrast', WNAME) - TBMAX
	g = cv2.getTrackbarPos('gamma', WNAME) - TBMAX

	print("b: " + str(b))

	cv2.convertScaleAbs(img, img2,  b)
	img = img2

cv2.destroyAllWindows()
