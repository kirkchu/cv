import cv2

# 讀取圖片
image = cv2.imread('data/cat.jpg')  # 替換為圖片的實際路徑

# 檢查圖片是否成功讀取
if image is None:
    print("無法讀取圖片，請檢查路徑")
else:
    h, w = image.shape[:2]

    def update(val):
        split = cv2.getTrackbarPos('Split', 'Image')
        img_disp = image.copy()
        if split > 0:
            left_half = img_disp[:, :split]
            left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
            left_gray_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            img_disp[:, :split] = left_gray_bgr
        cv2.imshow('Image', img_disp)

    cv2.namedWindow('Image')
    cv2.createTrackbar('Split', 'Image', w//2, w, update)
    update(w//2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
