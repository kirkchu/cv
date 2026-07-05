import cv2

image = cv2.imread('src/data/duck.jpg', 0)
feature = cv2.xfeatures2d.SIFT_create()
# feature = cv2.xfeatures2d.SURF_create()
feature = cv2.ORB_create(1)

kp = feature.detect(image)
print(list(kp))
print('keypoints: {}'.format(len(kp)))
image = cv2.drawKeypoints(
    image, kp, None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
