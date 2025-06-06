from ultralytics import YOLO
import cv2

image_path = 'data/test_lion.jpg'
model = YOLO('model/runs/classify/train/weights/best.pt')
results = model(image_path)[0]

top1_class_id = results.probs.top1  # 最可能的類別編號
class_name = results.names[top1_class_id]  # 類別名稱
confidence = results.probs.top1conf  # 信心值
# 顯示測試圖片的分類名稱與信心值
print(f'{class_name}({confidence:.2f})')

# 讀取圖片
img = cv2.imread(image_path)

# 在圖片上標示分類名稱與信心值
label = f"{class_name} ({confidence:.2f})"
# 取得文字大小以決定放置位置
text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
# 計算文字的座標位置 (左下角)
x = 10
y = 30

# 創建白色半透明背景
overlay = img.copy()
# 繪製白色背景框
cv2.rectangle(overlay, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), (255,255,255), -1)
# 設定透明度 (alpha 值: 0-1 之間，0 為完全透明，1 為不透明)
alpha = 0.7
# 將半透明背景與原圖合成
cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
# 繪製黑色文字
cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

# 顯示圖片
cv2.imshow("YOLO Classification Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()