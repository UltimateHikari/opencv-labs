# source: https://docs.opencv.org/3.1.0/d9/dc8/tutorial_py_trackbar.html
# https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/
import cv2
import numpy as np
import sys

WNAME = "Histogram equalization"
histsize = 256
histrange = (0,256)
hist_w = 512
hist_h = 400
bin_w = int(round( hist_w/histsize ))

if __name__ == '__main__': 
	img = cv2.imread(sys.argv[1])

	hist_h = img.shape[0]

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	res = cv2.equalizeHist(img)

	img_hist = cv2.calcHist(img, [0], None, [histsize], histrange, accumulate=False)
	img_chist = img_hist.copy()
	for i in range(1,histsize):
		img_chist[i] = img_chist[i] + img_chist[i-1]
	res_hist = cv2.calcHist(res, [0], None, [histsize], histrange, accumulate=False)
	res_chist = res_hist.copy()
	for i in range(1,histsize):
		res_chist[i] = res_chist[i] + res_chist[i-1]

	cv2.normalize(img_hist, img_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(img_chist, img_chist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(res_hist, res_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
	cv2.normalize(res_chist, res_chist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

	ih = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
	ich = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
	rh = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
	rch = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

	images = [ih, ich, rh, rch]
	histograms = [img_hist, img_chist, res_hist, res_chist]

	for image, histogram in zip(images, histograms):
		for i in range(1, histsize):
			cv2.line(image, ( bin_w*(i-1), hist_h - int(histogram[i-1]) ),
		(bin_w*(i), hist_h - int(histogram[i]) ), ( 255, 0, 0), thickness=2)

	cv2.namedWindow(WNAME)

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

	imghor = cv2.hconcat([img, images[0], images[1]])
	reshor = cv2.hconcat([res, images[2], images[3]])

	imgcon = cv2.vconcat([imghor, reshor])

	cv2.imshow(WNAME, imgcon)
	
	cv2.waitKey(0) 
	cv2.destroyAllWindows()
