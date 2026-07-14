#!/usr/bin/env python
"""MediaPipe 手勢辨識 SSE 串流伺服器。

使用 OpenCV 開啟攝影機，透過 MediaPipe GestureRecognizer 進行即時手勢辨識，
在畫面上繪製手掌地標與連線，並以 Server-Sent Events (SSE) 將畫面與辨識結果
傳送給前端網頁顯示。

參考: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer

啟動方式：
    python server.py

啟動後開啟瀏覽器至 http://localhost:8001 即可看到畫面。
"""
import base64
import json
import os
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import cv2
import mediapipe as mp

WEB_ROOT = os.path.dirname(os.path.abspath(__file__))
# 模型檔案放置於專案共用的 model 資料夾中
MODEL_PATH = os.path.join(WEB_ROOT, "..", "..", "model", "gesture_recognizer.task")

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 為了取得手掌地標連線資訊 (HAND_CONNECTIONS)
from mediapipe.tasks.python.vision import HandLandmarksConnections

# 由於畫面在偵測前已經過左右鏡像 (cv2.flip)，MediaPipe 判斷出的左右手
# 會與使用者實際舉起的手相反，因此需要對調顯示標籤才符合使用者直覺
HANDEDNESS_DISPLAY_MAP = {
    "Left": "Right",
    "Right": "Left",
}

# 內建手勢分類名稱 -> 中文顯示文字
GESTURE_LABELS = {
    "None": "未偵測到手勢",
    "Closed_Fist": "✊ 握拳",
    "Open_Palm": "🖐️ 張開手掌",
    "Pointing_Up": "☝️ 食指向上",
    "Thumb_Down": "👎 大拇指向下",
    "Thumb_Up": "👍 大拇指向上",
    "Victory": "✌️ 勝利手勢",
    "ILoveYou": "🤟 我愛你",
}

# 控制影像串流是否進行中的旗標
streaming_event = threading.Event()


def draw_hand_landmarks(image, hand_landmarks, handedness_label):
    """在影像上繪製單隻手的地標與連接線。"""
    # 左手用綠色、右手用橘紅色 (BGR)，區分辨識度更高
    line_color = (120, 220, 90) if handedness_label == "Left" else (60, 130, 255)
    joint_color = (0, 235, 255)
    h, w = image.shape[:2]

    if HandLandmarksConnections.HAND_CONNECTIONS:
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start_idx, end_idx = connection.start, connection.end
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(image, p1, p2, line_color, 2, cv2.LINE_AA)

    for landmark in hand_landmarks:
        p = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(image, p, 4, joint_color, -1, cv2.LINE_AA)
        cv2.circle(image, p, 5, (255, 255, 255), 1, cv2.LINE_AA)


def write_event(wfile, frame, hands_info, quality=85):
    """將影像與辨識結果編碼為 JSON，並以 SSE 格式送出。"""
    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return
    payload = {
        "image": base64.b64encode(buffer).decode("utf-8"),
        "hands": hands_info,
    }
    wfile.write(f"data:{json.dumps(payload)}\n\n".encode("utf-8"))
    wfile.flush()


class GestureHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/start"):
            self.handle_start()
        elif self.path.startswith("/stop"):
            self.handle_stop()
        else:
            super().do_GET()

    def handle_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        streaming_event.set()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            try:
                self.wfile.write(b'data:{"error":"camera_open_failed"}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 丟棄前幾幀，讓攝影機自動曝光與對焦穩定
        for _ in range(5):
            cap.read()

        if not os.path.exists(MODEL_PATH):
            try:
                self.wfile.write(b'data:{"error":"model_not_found"}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            cap.release()
            return

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        target_fps = 15
        frame_interval = 1.0 / target_fps

        try:
            with GestureRecognizer.create_from_options(options) as recognizer:
                while streaming_event.is_set():
                    loop_start = time.time()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # 左右鏡像處理，符合使用者直覺
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(time.time() * 1000)

                    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                    hands_info = []
                    if result.hand_landmarks:
                        for i, hand_landmarks in enumerate(result.hand_landmarks):
                            raw_handedness = (
                                result.handedness[i][0].category_name
                                if result.handedness and len(result.handedness) > i
                                else "Unknown"
                            )
                            handedness_label = HANDEDNESS_DISPLAY_MAP.get(raw_handedness, raw_handedness)
                            draw_hand_landmarks(frame, hand_landmarks, handedness_label)

                            gesture_name = "None"
                            gesture_score = 0.0
                            if result.gestures and len(result.gestures) > i and result.gestures[i]:
                                top_gesture = result.gestures[i][0]
                                gesture_name = top_gesture.category_name or "None"
                                gesture_score = float(top_gesture.score)

                            hands_info.append({
                                "handedness": handedness_label,
                                "gesture": gesture_name,
                                "label": GESTURE_LABELS.get(gesture_name, gesture_name),
                                "score": round(gesture_score, 3),
                            })

                    try:
                        write_event(self.wfile, frame, hands_info)
                    except (BrokenPipeError, ConnectionResetError):
                        # 前端已關閉連線，正常結束
                        break

                    elapsed = time.time() - loop_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            cap.release()

    def handle_stop(self):
        streaming_event.clear()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(b"data: stop\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(host="0.0.0.0", port=8001):
    handler = partial(GestureHandler, directory=WEB_ROOT)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving on http://{host}:{port} (root: {WEB_ROOT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        streaming_event.clear()
        server.server_close()


if __name__ == "__main__":
    main()
