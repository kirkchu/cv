import cv2
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGE_PATH = 'src/data/street.jpg'
MODEL_PATH = 'model/sam3.pt'

# 建立 SAM 3 predictor，設定信心閾值與模型路徑
overrides = dict(conf=0.3, task='segment', mode='predict', model=MODEL_PATH, save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 載入圖片（只載入一次，後續對話重複使用）
predictor.set_image(IMAGE_PATH)

print("=" * 60)
print("SAM 3 對話式偵測")
print(f"圖片: {IMAGE_PATH}")
print("輸入文字提示 (例如: chair, person, table, sofa) 來偵測物件")
print("輸入 'quit' 或 'exit' 結束程式")
print("=" * 60)

while True:
    try:
        text = input("\n請輸入提示文字: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n結束程式。")
        break

    if not text:
        continue

    if text.lower() in ('quit', 'exit', 'q'):
        print("結束程式。")
        break

    print(f"正在偵測: '{text}' ...")

    # 用文字提示找出對應物件
    results = predictor(text=[text])[0]

    # 顯示結果（畫上遮罩與框線）
    annotated = results.plot()
    window_name = f'SAM 3 - {text}'
    cv2.imshow(window_name, annotated)
    print("按任意鍵繼續對話...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

cv2.destroyAllWindows()