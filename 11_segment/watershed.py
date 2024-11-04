# idea from https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import time

WNAME1 = "Original"
WNAME2 = "Threshold"
WNAME3 = "Denoised Threshold"
WNAME4 = "Markers"
WNAME5 = "Watershed segmentation"

def imshow(wname, img):
	cv.namedWindow(wname)
	cv.imshow(wname, img)


#img = cv.imread('./hallx.jpg')
img = cv.imread(sys.argv[1])
c = int(sys.argv[2])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
channels = cv.split(img)
if (c > -1 and c < 3):
	print ("picking channel from 012 BGR: ", c)
	gray = channels[c]

imshow(WNAME1, gray)

# 4(gray) 3 1 5 2
k = int(sys.argv[3])
dn_iter = int(sys.argv[4])
bg_iter = int(sys.argv[5])
ratio = 0.1*int(sys.argv[6])

#Threshold Processing
#ret, imgtr = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#imgtr = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
imgtr = cv.Canny(gray, 100, 200)
imshow(WNAME2, imgtr)

# Denoise
# Actually now dilate-erode

kernel = np.ones((3, 3))
img_dilate = cv.dilate(imgtr, kernel, iterations=4)
imgdn = cv.erode(img_dilate, kernel, iterations=2)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
#imgdn = cv.morphologyEx(imgtr, cv.MORPH_OPEN, kernel, iterations=dn_iter)
imshow(WNAME3, imgdn)

# Sure zones

sure_bg = cv.dilate(imgdn, kernel, iterations=bg_iter)
imshow('Sure Background', sure_bg)

dist = cv.distanceTransform(imgdn, cv.DIST_L2, 5)
imshow('Distance Transform', dist)

ret, sure_fg = cv.threshold(dist, ratio * dist.max(), 255, cv.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)  
imshow('Sure Foreground', sure_fg)
 
unknown = cv.subtract(sure_bg, sure_fg)
imshow('Unknown', unknown)

# Marker labelling

ret, markers = cv.connectedComponents(sure_fg)
 
# bg = 1 unk = 0
markers += 1
markers[unknown == 255] = 0

time.sleep(1)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
fig.canvas.draw()
markers_plot = np.array(fig.canvas.renderer.buffer_rgba())

cv.namedWindow(WNAME4)
cv.imshow(WNAME4, cv.cvtColor(markers_plot, cv.COLOR_RGBA2BGR))

# finally watershedding

markersw = cv.watershed(img, markers)
 
figw, axw = plt.subplots(figsize=(5, 5))
axw.imshow(markersw, cmap="tab20b")
axw.axis('off')
figw.canvas.draw()
markers_plotw = np.array(figw.canvas.renderer.buffer_rgba())

cv.namedWindow(WNAME5)
cv.imshow(WNAME5, cv.cvtColor(markers_plotw, cv.COLOR_RGBA2BGR))

# draw contours
labels = np.unique(markers)
 
objs = []
for label in labels[2:]:  
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    contours, hierarchy = cv.findContours(
        target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    objs.append(contours[0])
 
# Draw the outline
img = cv.drawContours(img, objs, -1, color=(0, 0, 255), thickness=2)
imshow("Contoured", img)

while True:
	k = cv.waitKey(0)
	if k == 27:
		break


cv.destroyAllWindows()

