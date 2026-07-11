import cv2
from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = 'model/sam3.pt'
CAMERA_INDEX = 0

# 建立 SAM 3 predictor，設定信心閾值與模型路徑
overrides = dict(conf=0.25, task='segment', mode='predict', model=MODEL_PATH, save=False, device='mps', quantize=16)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 開啟攝影機
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError('無法開啟攝影機')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        # 將目前影格設為 predictor 的輸入圖片
        predictor.set_image(frame)

        # 用文字提示 "watch" 找出手錶
        results = predictor(text=['watch'])[0]

        # 顯示結果 (畫上遮罩與框線)
        annotated = results.plot()
        cv2.imshow('SAM 3 - watch', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
