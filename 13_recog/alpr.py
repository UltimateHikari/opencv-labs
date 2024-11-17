import wget
import os
import sys
import imutils
import cv2 as cv
import pytesseract

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

WNAME1 = "Original"
WNAME2 = "Plate contour"
WNAME3 = "Plate"
plcolor = (0,255,0)
plsize = 2

path = 'haarcascade_russian_plate_number.xml'

if (not os.path.isfile(path)):
	wget.download('https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_russian_plate_number.xml')

lp_cascade = cv.CascadeClassifier(path)

img = cv.imread(sys.argv[1])
cv.imshow(WNAME1, img)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
numbers = lp_cascade.detectMultiScale(img, scaleFactor=1.05)

plates_gray = []

for (x,y,w,h) in numbers:
	plates_gray.append(img_gray[y:y+h, x:x+w])
	cv.rectangle(img, (x, y), (x + w, y + h), plcolor, plsize)

cv.imshow(WNAME2, img)
print ("number of detected plates: ", len(plates_gray))
	
for i, plate in enumerate(plates_gray):
	text = pytesseract.image_to_string(plates_gray[i], config='-l eng --psm 7 --oem 1')
	print (text)
	plate_r = imutils.resize(plates_gray[i], width=800)
	cv.putText(plate_r, cleanup_text(text), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), plsize, cv.LINE_AA)
	if (len(cleanup_text(text)) > 5):
		cv.imshow(WNAME3+"-"+str(i), plate_r)

ch = cv.waitKey(0)

cv.destroyAllWindows()




