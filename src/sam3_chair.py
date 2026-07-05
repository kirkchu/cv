"""
SAM 3 基本範例：用文字提示 "chair" 找出 src/chair.jpg 中的椅子並畫出遮罩。

事前準備：
1. pip install -U ultralytics   (需要 ultralytics>=8.3.237 才有 SAM 3 支援)
2. 到 https://huggingface.co/facebook/sam3 申請權重存取，下載 sam3.pt
3. 將 sam3.pt 放到 model/sam3.pt (或修改下面 MODEL_PATH)
4. 準備 src/chair.jpg (要偵測椅子的圖片)
"""

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGE_PATH = 'src/data/bus.jpg'
MODEL_PATH = 'model/sam3.pt'

# 建立 SAM 3 predictor，設定信心閾值與模型路徑
overrides = dict(conf=0.25, task='segment', mode='predict', model=MODEL_PATH, save=True)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 載入圖片
predictor.set_image(IMAGE_PATH)

# 用文字提示 "chair" 找出所有椅子
# results = predictor(text=['chair'])[0]
results = predictor(text=['bus'])[0]

# 顯示結果 (畫上遮罩與框線)
annotated = results.plot()
cv2.imshow('SAM 3 - chair', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
