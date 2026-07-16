import cv2

image = cv2.imread('data/polygon.png', 0)
edged = cv2.Canny(image, 30, 150)
contours, hierarchy = cv2.findContours(
    edged, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

# cnt = contours[0]
cnt = contours[1]
# cnt = contours[2]

cnt = cv2.approxPolyDP(cnt, 10, closed=True)
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
if defects is None:
    defects = []
print('凸點數量：{}'.format(len(hull)))
print('凹點數量：{}'.format(len(defects)))

cv2.imshow('frame', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
