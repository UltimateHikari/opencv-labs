import sys
import cv2 as cv

WNAME1 = "Train"
WNAME2 = "Query"
WNAME3 = "Result"
WNAME4 = "Noised"
WNAME5 = "NoisedResult"
plcolor = (0,255,0)
plcolor2 = (0,0,255)
plsize = 2

def match_template_bb(train, query):
	result = cv.matchTemplate(train, query,
        cv.TM_CCOEFF_NORMED)
	(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
	print(minVal, maxVal, minLoc, maxLoc)

	(startX, startY) = maxLoc
	endX = startX + query.shape[1]
	endY = startY + query.shape[0]
	return (startX, startY, endX, endY)

# ./train.png ./query.png
train = cv.imread(sys.argv[1])
#cv.imshow(WNAME1, train)

query = cv.imread(sys.argv[2])
cv.imshow(WNAME2, query)

# noise type: 0 - randu, 1 - randn
t = int(sys.argv[3])
# noise param 1 (mean)
m = int(sys.argv[4])
# noise param 2 (sigma or high)
s = int(sys.argv[5])
# rotation degree 15/32/45
d = int(sys.argv[6])
# resize float 1.5
r = float(sys.argv[7])
print(r)

train_gray = cv.cvtColor(train, cv.COLOR_BGR2GRAY)
query_gray = cv.cvtColor(query, cv.COLOR_BGR2GRAY)


(startX, startY, endX, endY) = match_template_bb(train_gray, query_gray)

res1 = train.copy()
cv.rectangle(res1, (startX, startY), (endX, endY), plcolor, plsize)
cv.imshow(WNAME3, res1)

#=== noise incoming ===#

mv = (m,m,m)
sv = (s,s,s)

if t > 0:
        # add gaussian
        noise = cv.randn(train.copy(), mv, sv) 
else:
        noise = cv.randu(train.copy(), mv, sv)

img_res = cv.add(train.copy(), noise)

(h, w) = img_res.shape[:2]
(cX, cY) = (w // 2, h // 2)

M = cv.getRotationMatrix2D((cX, cY), d, 1.0)
img_rot = cv.warpAffine(img_res, M, (w, h))

img_res = cv.resize(img_rot, ((int)(w*r), (int)(h*r)))

img_gray = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)
#cv.imshow(WNAME4, img_gray)

(startX2, startY2, endX2, endY2) = match_template_bb(img_gray, query_gray)

cv.rectangle(img_res, (startX2, startY2), (endX2, endY2), plcolor2, plsize)
cv.imshow(WNAME5, img_res)

ch = cv.waitKey(0)

cv.destroyAllWindows()




