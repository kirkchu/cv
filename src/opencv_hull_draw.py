import cv2

image = cv2.imread('src/data/polystar.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 150)
contours, hierarchy = cv2.findContours(
    edged, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

cnt = cv2.approxPolyDP(cnt, 30, closed=True)
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
if defects is None:
    defects = []

print('凸點數量：{}'.format(len(hull)))
print('凹點數量：{}'.format(len(defects)))

for i in range(defects.shape[0]):
    s,e,f,d = defects[i].flatten()
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(image, start, end, [0, 255, 0], 2)
    cv2.circle(image, far, 10, [0, 0, 255], -1, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
