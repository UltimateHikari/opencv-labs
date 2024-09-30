import cv2 as cv
import sys
import random
import numpy as np


WNAME = "Additive Noise Lena"

def additive_noise(img, k):
	res = img.copy()
	height, width = res.shape
	for y in range(height):
		for x in range(width):
			r = random.randint(-1*k, k)
			res[y,x] = np.uint8(np.clip(int(res[y,x])+r,0,255))
	return res

def denoise(images):
	height, width = images[0].shape
	res = np.full((height, width), 0, dtype=np.uint8)
	for y in range(height):
		for x in range(width):
			pixel_value = 0
			for img in images:
				pixel_value += img[y,x]
			res[y,x] = int(pixel_value/len(images))
	return res



k = int(sys.argv[2])
N = int(sys.argv[3])
if N < 2:
	N = 2

#img = cv.imread('./Lena512.png')
img = cv.imread(sys.argv[1])

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#img1 = additive_noise(img_gray,k)
#img2 = additive_noise(img_gray,k)
#img3 = additive_noise(img_gray,k)

images = []

print("Generating..")

for i in range(N):
	images.append(additive_noise(img_gray,k))
print(len(images))

print("Done.\nDenoising...")
img_res = denoise(images)

cv.namedWindow(WNAME)

cv.imshow(WNAME, cv.hconcat([img_gray, img_res, images[0], images[1], images[2]]))

cv.waitKey(0)

cv.destroyAllWindows()
