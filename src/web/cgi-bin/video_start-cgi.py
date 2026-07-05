#!/Users/ckk/venv/cv/bin/python
import cv2
import base64
import sys

class StreamingEvent:
    def __init__(self):
        print('Content-Type: text/event-stream\n\n')

    def write(self, frame):
        data = cv2.imencode('.jpg', frame)[1].tobytes()
        base64_encode = base64.b64encode(data).decode('utf-8')
        print(f'data:{base64_encode}\n')
        sys.stdout.flush()

streaming = StreamingEvent()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.flip(frame, 1)
    streaming.write(frame)