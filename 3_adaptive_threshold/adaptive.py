import cv2 as cv
import sys

#img = cv.imread('./src.jpg')
img = cv.imread(sys.argv[1])
grayscaled = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

th = cv.adaptiveThreshold(grayscaled, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 115, 1)
sth = cv.adaptiveThreshold(grayscaled, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)
gth = cv.adaptiveThreshold(grayscaled, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
cv.imshow('original',img)
cv.imshow('Adaptive mean threshold',th)
cv.imshow('Adaptive small gaussian threshold',sth)
cv.imshow('Adaptive gaussian threshold',gth)
cv.waitKey(0)
cv.destroyAllWindows()
