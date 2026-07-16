import cv2

# 讀取圖片
image = cv2.imread('data/lane.jpg')  # 替換為圖片的實際路徑

# 檢查圖片是否成功讀取
if image is None:
    print("無法讀取圖片，請檢查路徑")
else:
    # 轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def on_trackbar(val):
        # 取得 trackbar 的值作為閾值
        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Image', binary)
    
    # 建立視窗與 trackbar
    cv2.namedWindow('Binary Image')
    cv2.createTrackbar('Threshold', 'Binary Image', 200, 255, on_trackbar)
    # 初始化顯示
    on_trackbar(200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
