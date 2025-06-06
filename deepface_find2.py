from deepface import DeepFace
import cv2

# 設定來源圖片與資料集資料夾
source_img = "face/find_test1.jpg"
db_path = "face/database"

# 執行人臉辨識
results = DeepFace.find(img_path=source_img, db_path=db_path)

# 根據結果輸出 identity 或 ❌
if len(results) > 0 and not results[0].empty:
    identity = results[0].iloc[0]['identity']
    parts = identity.split("/")
    if len(parts) >= 3:
        label = parts[2]
        img = cv2.imread(source_img)
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(identity)
else:
    print("❌")
