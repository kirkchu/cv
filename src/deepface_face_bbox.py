from deepface import DeepFace
import cv2

# 讀取圖片
img_path = "src/face/face_voyager.jpg"
img = cv2.imread(img_path)

# 使用 deepface 偵測人臉
detections = DeepFace.extract_faces(
    img_path=img_path, detector_backend='retinaface')

print(detections)
# 畫出 bounding box 與臉部地標
for face in detections:
    area = face['facial_area']
    x, y, w, h = area['x'], area['y'], area['w'], area['h']
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    left_eye = area.get('left_eye')
    right_eye = area.get('right_eye')
    nose = area.get('nose')
    mouth_left = area.get('mouth_left')
    mouth_right = area.get('mouth_right')

    if left_eye:
        cv2.circle(img, left_eye, 3, (255, 255, 255), -1)  # 白色：左眼
    if right_eye:
        cv2.circle(img, right_eye, 3, (255, 0, 0), -1)  # 藍色：右眼
    if nose:
        cv2.circle(img, nose, 3, (0, 255, 255), -1)  # 黃色：鼻子
    if mouth_left and mouth_right:
        cv2.line(img, mouth_left, mouth_right, (0, 0, 255), 2)  # 紅色：嘴巴

# 顯示結果到螢幕上
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
