import cv2
import mediapipe as mp
import numpy as np

# 初始化 Mediapipe 的 Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 讀取圖檔
image_path = "face/face.jpg"  # 替換為你的圖檔路徑
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 將 BGR 圖片轉換為 RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 偵測人臉地標
results = face_mesh.process(rgb_image)

# 繪製地標
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        points = []
        for idx, landmark in enumerate(face_landmarks.landmark):
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # 繪製最外圍的白色線條
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(image, [hull], isClosed=True, color=(255, 255, 255), thickness=2)

# 顯示結果
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
