import sys
import cv2 as cv
import imutils

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# yt-dlp -f 136 https://youtu.be/GJNjaRJWVP8?si=STQl187QLl1B-Qia --username=oauth --password=""
# ~/Videos/The CCTV People Demo 2 [GJNjaRJWVP8].mp4
vid = cv.VideoCapture(sys.argv[1])
# 400
minwid = int(sys.argv[2])

bcolor_low = (20, 0, 250)
bcolor_med = (0, 127, 255)
bcolor_high = (20, 255, 0)

bsize = 2
WNAME = "Pedestrians"

prob_low = 0.13
prob_med = 0.3
prob_high = 0.7

st = 2
pd = 8
sc = 1.04

cv.namedWindow(WNAME)

while vid.isOpened():
	ret, image = vid.read()
	if ret:
		image = imutils.resize(image, width=min(minwid, image.shape[1]))
		
		img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		(regions, weights) = hog.detectMultiScale(
				img_gray,
				winStride=(st, st),
				padding=(pd, pd),
				scale=sc
				)

		for i, (x, y, w, h) in enumerate(regions):
			prob = weights[i]
			bcolor = bcolor_high
			match prob:
				case prob if prob < prob_low:
					continue
				case prob if prob_low <= prob < prob_med:
					bcolor = bcolor_low
				case prob if prob_med <= prob < prob_high:
					bcolor = bcolor_med
			
			cv.rectangle(image, (x, y), (x + w, y + h), bcolor, bsize)

		cv.imshow(WNAME, image)
		if cv.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

vid.release()
cv.destroyAllWindows()
