import cv2
from deepface import DeepFace

# 設定人臉資料庫路徑
database_path = "face/database"

# 開啟攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 水平翻轉畫面
    if not ret:
        break

    # 辨識人臉
    try:
        result = DeepFace.find(img_path = frame, db_path = database_path, enforce_detection=False, silent=True, anti_spoofing=True)
        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            parts = identity.split('/')
            name = parts[2] if len(parts) > 2 else identity
            cv2.putText(frame, f"Match: {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    except Exception as e:
        cv2.putText(frame, "Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("DeepFace Video Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
