import cv2
import mediapipe as mp
import numpy as np
import time
url = 'https://cctv-ss04.thb.gov.tw/T14A-d61a0c91'
# 初始化姿勢偵測模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 設定姿勢偵測模型配置
pose = mp_pose.Pose(
    static_image_mode=False,          # 是否為靜態圖像模式
    model_complexity=1,               # 模型複雜度 (0, 1, 2)
    smooth_landmarks=True,            # 是否平滑關鍵點
    min_detection_confidence=0.5,     # 最低檢測信心度
    min_tracking_confidence=0.5       # 最低追蹤信心度
)

# 打開攝影機 (或讀取影片)
cap = cv2.VideoCapture(url)  # 使用 0 代表預設攝影機，若要讀取影片文件，請指定路徑

# 計算 FPS 的變數
prev_time = 0
curr_time = 0

while cap.isOpened():
    # 讀取一幀影像
    success, image = cap.read()
    if not success:
        print("無法讀取攝影機或影片。")
        break
        
    # 為提高效能，可選擇性地將影像標記為不可寫入
    image.flags.writeable = False
    
    # 將 BGR 轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 進行姿勢偵測
    results = pose.process(image_rgb)
    
    # 將影像標記為可寫入以便繪製
    image.flags.writeable = True
    
    # 轉回 BGR 格式以便 OpenCV 顯示
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 繪製姿勢關鍵點
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # 範例：獲取特定關鍵點 (例如右手腕)
        if results.pose_landmarks.landmark:
            h, w, c = image.shape
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_wrist_pos = (int(right_wrist.x * w), int(right_wrist.y * h))
            
            # 在右手腕位置畫一個圓
            cv2.circle(image, right_wrist_pos, 10, (0, 255, 0), -1)
    
    # 計算並顯示 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 顯示輸出影像
    cv2.imshow('MediaPipe Pose', image)
    
    # 按下 'q' 鍵退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
