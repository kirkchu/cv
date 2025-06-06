from deepface import DeepFace
import cv2

# 讀取圖片
img_path = "face/face_voyager.jpg"
img = cv2.imread(img_path)

# 使用 deepface 偵測人臉
detections = DeepFace.extract_faces(
    img_path=img_path, detector_backend='retinaface')

# 畫出 bounding box
for face in detections:
    x, y, w, h, _, _ = face['facial_area'].values()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 顯示結果到螢幕上
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
