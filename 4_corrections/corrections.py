# source: https://docs.opencv.org/3.1.0/d9/dc8/tutorial_py_trackbar.html
# https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/
import cv2
import numpy as np
import sys

def nothing(x):
	pass

CMAX = 127
BMAX = 255
WNAME = 'Brightness, Contrast, Gamma'

def bsg(event=0): 
	b = cv2.getTrackbarPos('brightness', WNAME)
	c = cv2.getTrackbarPos('contrast', WNAME)
	g = cv2.getTrackbarPos('gamma, (* 0.01)', WNAME)
	effect = controller(img, b, c, g) 
	print(b,c,g)
	cv2.imshow(WNAME, effect)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	print(invGamma, gamma)
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def controller(img, brightness=255, 
               contrast=127, gamma = 100):

	brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
	gamma = gamma * 0.01

	if brightness != 0: 
		if brightness > 0: 
			shadow = brightness 
			max = BMAX
		else: 
			shadow = 0
			max = BMAX + brightness 
		al_pha = (max - shadow) / 255
		ga_mma = shadow 
		cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma) 
	else: 
		cal = img 

	if contrast != 0: 
		Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
		Gamma = 127 * (1 - Alpha) 
		cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

	gamma_img = adjust_gamma(cal, gamma)
	
	return gamma_img 

if __name__ == '__main__': 
	img = cv2.imread(sys.argv[1])
	
	cv2.namedWindow(WNAME)

	cv2.imshow(WNAME, img)
	
	cv2.createTrackbar('brightness', WNAME, BMAX, BMAX*2, bsg)
	cv2.createTrackbar('contrast', WNAME, CMAX, CMAX*2, bsg)
	cv2.createTrackbar('gamma, (* 0.01)', WNAME, 100, 300, bsg)
	bsg(0)

	cv2.waitKey(0) 
	cv2.destroyAllWindows()
