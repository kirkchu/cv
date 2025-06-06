import cv2
import mediapipe as mp
import time
import numpy as np # numpy is used by mp.Image

# 從 mediapipe.tasks.python 匯入必要的模組
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult # 主要用於類型提示，此處未使用回呼
VisionRunningMode = mp.tasks.vision.RunningMode

# 為了取得 HAND_CONNECTIONS
mp_hands = mp.solutions.hands

# 手部地標模型檔案的路徑
# 請從 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task 下載
MODEL_PATH = 'model/hand_landmarker.task' 

def main():
    try:
        # 建立 HandLandmarkerOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO, # 使用 VIDEO 模式進行即時影像流處理
            num_hands=2,                         # 最多偵測 2 隻手
            min_hand_detection_confidence=0.5,   # 手部偵測的最小信心度
            min_hand_presence_confidence=0.5,    # 手部存在的最小信心度 (用於追蹤)
            min_tracking_confidence=0.5)         # 手部追蹤的最小信心度

        # 使用 'with' 陳述式建立 HandLandmarker，確保資源被正確釋放
        with HandLandmarker.create_from_options(options) as landmarker:
            # 開啟攝影機
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("無法開啟攝影機")
                return

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("忽略空的攝影機影像幀。")
                    continue

                # 1. 影像讀取後先左右鏡像處理
                image = cv2.flip(image, 1)

                # 2. 將 BGR 影像轉換為 RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 3. 建立 MediaPipe Image 物件
                # MediaPipe Image 需要 NumPy 陣列
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # 4. 獲取當前時間戳 (毫秒)
                frame_timestamp_ms = int(time.time() * 1000)

                # 5. 進行手部地標偵測 (同步呼叫)
                hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # 6. 處理偵測結果並在影像上繪製 (在原始 BGR 影像上操作)
                annotated_image = image.copy()
                if hand_landmarker_result.hand_landmarks:
                    for i in range(len(hand_landmarker_result.hand_landmarks)):
                        hand_landmarks = hand_landmarker_result.hand_landmarks[i]
                        handedness = hand_landmarker_result.handedness[i][0].category_name

                        # 根據左右手設定顏色 (OpenCV 使用 BGR 格式)
                        if handedness == "Left":
                            hand_color = (0, 255, 0)  # 綠色
                        else:  # Right
                            hand_color = (0, 0, 255)  # 紅色
                        
                        joint_color = (0, 255, 255) # 黃色 (用於關節點)

                        # 繪製手部連接線
                        if mp_hands.HAND_CONNECTIONS:
                            for connection in mp_hands.HAND_CONNECTIONS:
                                start_idx = connection[0]
                                end_idx = connection[1]

                                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                                    start_landmark = hand_landmarks[start_idx]
                                    end_landmark = hand_landmarks[end_idx]

                                    # 將正規化座標轉換為像素座標
                                    start_point = (int(start_landmark.x * annotated_image.shape[1]),
                                                   int(start_landmark.y * annotated_image.shape[0]))
                                    end_point = (int(end_landmark.x * annotated_image.shape[1]),
                                                 int(end_landmark.y * annotated_image.shape[0]))
                                    
                                    cv2.line(annotated_image, start_point, end_point, hand_color, 2)

                        # 繪製關節點 (黃色實心圓) 並標上編號
                        for idx, landmark in enumerate(hand_landmarks):
                            x = int(landmark.x * annotated_image.shape[1])
                            y = int(landmark.y * annotated_image.shape[0])
                            
                            # 繪製黃色實心圓
                            cv2.circle(annotated_image, (x, y), 5, joint_color, -1) 
                            
                            # 標上編號 (白色文字)
                            cv2.putText(annotated_image, str(idx), (x + 8, y - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # 顯示處理後的影像
                cv2.imshow('MediaPipe 手部地標偵測', annotated_image)

                # 按下 'q' 鍵退出
                if cv2.waitKey(1) == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
