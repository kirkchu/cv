import cv2
import numpy as np

# 讀取圖片
image = cv2.imread('data/erode_dilate.jpg')  # 替換為圖片的實際路徑

# 檢查圖片是否成功讀取
if image is None:
    print("無法讀取圖片，請檢查路徑")
else:
    # 建立3x3的結構元素
    kernel = np.ones((3, 3), np.uint8)
    # 侵蝕處理，疊代10次
    eroded = cv2.erode(image, kernel, iterations=20)
    # 膨脹處理，疊代10次
    dilated = cv2.dilate(image, kernel, iterations=20)
    # 水平合併
    merged = np.hstack((image, eroded, dilated))
    # 顯示合併後的圖
    cv2.imshow('Original | Eroded | Dilated', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
