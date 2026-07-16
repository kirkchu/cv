import cv2

image = cv2.imread('data/duck.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def update_features(n_features):
    n_features = max(1, n_features)  # 避免為0
    feature = cv2.ORB_create(n_features)
    kp = feature.detect(gray)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    feature_image = cv2.drawKeypoints(
        img_bgr, kp, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow('image', feature_image)

cv2.namedWindow('image')
cv2.createTrackbar('n_features', 'image', 1, 100, update_features)
update_features(1)  # 初始化顯示

cv2.waitKey(0)
cv2.destroyAllWindows()
