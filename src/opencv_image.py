import cv2

# 讀取圖片
image = cv2.imread('data/cat.jpg')  # 替換為圖片的實際路徑

# 檢查圖片是否成功讀取
if image is None:
    print("無法讀取圖片，請檢查路徑")
else:
    # 顯示圖片
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # 等待按鍵輸入
    cv2.destroyAllWindows()
