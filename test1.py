import cv2

# 讀取圖片
image = cv2.imread('data/duck.jpg')

# 檢查圖片是否成功讀取
if image is None:
    print('無法讀取圖片，請檢查路徑')
else:
    cv2.imshow('duck', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
