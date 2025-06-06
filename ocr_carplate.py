import easyocr
import cv2
import numpy as np

img = cv2.imread('data/carplate.jpg')
reader = easyocr.Reader(['en'])
results = reader.readtext(img, adjust_contrast=0.1, min_size=50)

for (bbox, text, confidence) in results:
    # bbox 是四個點的 list: [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
    # 將座標轉換為整數 tuple 的 NumPy 陣列
    pts = np.array([list(map(int, point)) for point in bbox], dtype=np.int32)
    # 繪製多邊形框
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # 在 bounding box 左上角顯示文字 (轉換為大寫)
    text_position = tuple(pts[0]) # 使用第一個點作為文字位置的基準
    cv2.putText(img, text.upper(), (text_position[0], text_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # 輸出轉換為大寫的文字
    print(f"Detected text: {text.upper()} (Confidence: {confidence:.2f})")

# 顯示結果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
