import cv2 as cv
import sys

percent = int(sys.argv[2])

#img = cv.imread('/home/andy/Pictures/wall.jpg')
img = cv.imread(sys.argv[1])

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

height, width = img_gray.shape

result = 0

for y in range(height):
	for x in range(width):
		if img_gray[y,x] > percent:
			result += 1

print(result/(height*width))
cv.imshow('gray', img_gray)
cv.waitKey(0)

cv.imshow('orig', img)
cv.waitKey(0)

cv.destroyAllWindows()
