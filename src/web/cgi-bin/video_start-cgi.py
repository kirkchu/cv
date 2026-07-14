#!/Users/ckk/github/cv/.venv/bin/python
import os
import sys

# 強制 stdout 行緩衝或無緩衝，確保每一幀都能即時送到前端
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

import base64
import time

import cv2


def send_sse_headers():
    """輸出 SSE 所需的 HTTP 表頭。"""
    sys.stdout.write("Content-Type: text/event-stream\r\n")
    sys.stdout.write("Cache-Control: no-cache\r\n")
    sys.stdout.write("Connection: keep-alive\r\n")
    sys.stdout.write("Access-Control-Allow-Origin: *\r\n")
    sys.stdout.write("\r\n")
    sys.stdout.flush()


def write_frame(frame, quality=85):
    """將影像編碼為 JPEG base64，並以 SSE 格式送出。"""
    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return
    base64_data = base64.b64encode(buffer).decode("utf-8")
    sys.stdout.write(f"data:{base64_data}\n\n")
    sys.stdout.flush()


def main():
    send_sse_headers()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.stdout.write("data:camera_open_failed\n\n")
        sys.stdout.flush()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 丟棄前幾幀，讓攝影機自動曝光與對焦穩定
    for _ in range(5):
        cap.read()

    target_fps = 15
    frame_interval = 1.0 / target_fps

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.resize(frame, (320, 240))
            frame = cv2.flip(frame, 1)

            try:
                write_frame(frame)
            except BrokenPipeError:
                # 前端已關閉連線，正常結束
                break

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cap.release()


if __name__ == "__main__":
    main()