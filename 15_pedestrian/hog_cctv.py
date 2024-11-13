import sys
import cv2 as cv
import imutils

from multiprocessing.pool import ThreadPool
from collections import deque

from common import clock, draw_str, StatValue
import video

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# yt-dlp -f 136 https://youtu.be/GJNjaRJWVP8?si=STQl187QLl1B-Qia --username=oauth --password=""
# ~/Videos/The CCTV People Demo 2 [GJNjaRJWVP8].mp4
cap = cv.VideoCapture(sys.argv[1])
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

preset = sys.argv[3]

st = 4
pd = 8
sc = 1.05

if preset == "slow":
	st = 2
	sc = 1.04

def process_frame(frame, t0):
	original_width = frame.shape[1]
	frame = imutils.resize(frame, width=min(minwid, frame.shape[1]))
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	(regions, weights) = hog.detectMultiScale(
			frame_gray,
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
	
		cv.rectangle(frame, (x, y), (x + w, y + h), bcolor, bsize)

	frame = imutils.resize(frame, width=original_width)

	return frame, t0

cv.namedWindow(WNAME)

#threadn = cv.getNumberOfCPUs() - 4
threadn = 6
print("Threads: ", threadn)
pool = ThreadPool(processes = threadn)
pending = deque()

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()

while True:
	while len(pending) > 0 and pending[0].ready():
		res, t0 = pending.popleft().get()
		latency.update(clock() - t0)
		
		draw_str(res, (20, 20), "latency        :  %.1f ms" % (latency.value*1000))
		draw_str(res, (20, 40), "frame interval :  %.1f ms" % (frame_interval.value*1000))
		
		cv.imshow(WNAME, res)

	if len(pending) < threadn:
		_ret, frame = cap.read()
		t = clock()
		frame_interval.update(t - last_frame_time)
		last_frame_time = t
			
		task = pool.apply_async(process_frame, (frame.copy(), t))
		pending.append(task)
		
	ch = cv.waitKey(1)
	if ch & 0xFF == ord('q'):
		break

while len(pending) > 0:
	if pending[0].ready:
		res, t0 = pending.popleft().get()
	else:
		time.sleep(1)
		

cap.release()
cv.destroyAllWindows()
