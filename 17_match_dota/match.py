import sys
import os
import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression

# https://stackoverflow.com/a/72863718
# max-supress and return only left-top x position for each result
def max_supression(result, template):
	threshold = 0.90
	(yCoords, xCoords) = np.where(result >= threshold)

	# Perform non-maximum suppression.
	template_h, template_w = template.shape[:2]
	rects = []
	for (x, y) in zip(xCoords, yCoords):
		rects.append((x, y, x + template_w, y + template_h))
	return [ a[0] for a in non_max_suppression(np.array(rects))]


# ./train.png ./query.png
train = cv.imread(sys.argv[1])
train_gray = cv.cvtColor(train, cv.COLOR_BGR2GRAY)

imagenames = ['zero.png', 'one.png', 'two.png', 'three.png', 'four.png', 'five.png', 'six.png', 'seven.png', 'eight.png', 'nine.png']
images = [cv.imread(imagename, cv.IMREAD_GRAYSCALE) for imagename in imagenames]

#for i, image in enumerate(images):
#	cv.imshow("Image"+str(i), image)

results = [(i, cv.matchTemplate(train_gray, image, cv.TM_CCOEFF_NORMED), image) for (i, image) in enumerate(images)]

tresults = [(i, max_supression(result, image)) for (i, result, image) in results]

numbers = []

for (i, result) in tresults:
	#print(i, result)
	for j in result:
		numbers.append((j,i))

numbers.sort()
onlynumbers = [str(i) for (j,i) in numbers]
index = 0
inserted = 0
while (index - 2) > len(numbers) * (-1):
	index = index - 2
	onlynumbers.insert(index-inserted, ':')
	inserted += 1

#print(onlynumbers)
print(''.join(onlynumbers))

cv.imshow("Reference", train)
ch = cv.waitKey(0)
cv.destroyAllWindows()

