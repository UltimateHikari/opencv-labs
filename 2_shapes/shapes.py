import cv2 as cv
import sys
import numpy as np

# python3 shapes.py ./cat.png ./new-cat.png

img = cv.imread(sys.argv[1])

cv.circle(img, (580, 160), 20, (255, 0, 0), 5)

cv.circle(img, (580, 75), 5, (255, 0, 0), 5)
cv.circle(img, (560, 40), 5, (255, 0, 0), 5)
cv.circle(img, (540, 25), 5, (255, 0, 0), 5)

x = np.array([540, 560, 580])
y = np.array([25,40,75])

z = np.polyfit(x, y, 2)
lspace = np.linspace(540,580,100)

print(lspace)

draw_x = lspace
draw_y = np.polyval(z, draw_x)

draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)

cv.polylines(img, [draw_points], False, (128,255,128), 5)

cv.imshow('orig', img)
cv.waitKey(0)

cv.imwrite(sys.argv[2], img)

cv.destroyAllWindows()
