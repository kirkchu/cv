#!/usr/bin/env python
"""合併 video_start-cgi.py 與 video_stop-cgi.py 的功能，並內建 web server。

啟動方式：
    python server.py

啟動後不需要再使用 `python -m http.server --cgi`。
"""
import base64
import os
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import cv2

WEB_ROOT = os.path.dirname(os.path.abspath(__file__))

# 控制影像串流是否進行中的旗標
streaming_event = threading.Event()


def write_frame(wfile, frame, quality=85):
    """將影像編碼為 JPEG base64，並以 SSE 格式送出。"""
    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return
    base64_data = base64.b64encode(buffer).decode("utf-8")
    wfile.write(f"data:{base64_data}\n\n".encode("utf-8"))
    wfile.flush()


class CameraHandler(SimpleHTTPRequestHandler):
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
                self.wfile.write(b"data:camera_open_failed\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 丟棄前幾幀，讓攝影機自動曝光與對焦穩定
        for _ in range(5):
            cap.read()

        target_fps = 15
        frame_interval = 1.0 / target_fps

        try:
            while streaming_event.is_set():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.resize(frame, (320, 240))
                frame = cv2.flip(frame, 1)

                try:
                    write_frame(self.wfile, frame)
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


def main(host="0.0.0.0", port=8000):
    handler = partial(CameraHandler, directory=WEB_ROOT)
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
