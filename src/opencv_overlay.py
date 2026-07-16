import cv2
import numpy as np

# 讀取底圖與要疊加的圖片
background = cv2.imread('data/beach.jpg')
overlay = cv2.imread('data/lantern.png', cv2.IMREAD_UNCHANGED)

if background is None:
    print("無法讀取 data/beach.jpg")
    exit(1)
if overlay is None:
    print("無法讀取 data/lantern.png")
    exit(1)

# 取得底圖尺寸
bg_h, bg_w = background.shape[:2]

# 將天燈圖縮放成與底圖相同大小
overlay_resized = cv2.resize(overlay, (bg_w, bg_h), interpolation=cv2.INTER_AREA)

# 處理透明通道（RGBA）
if overlay_resized.shape[2] == 4:
    # 分離 BGR 與 Alpha
    overlay_bgr = overlay_resized[:, :, :3]
    alpha = overlay_resized[:, :, 3] / 255.0

    # 使用 alpha 混合
    for c in range(3):
        background[:, :, c] = (alpha * overlay_bgr[:, :, c] + (1 - alpha) * background[:, :, c]).astype(np.uint8)
else:
    # 無透明通道，直接覆蓋
    background = overlay_resized

# 顯示結果
cv2.imshow('Overlay Result', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存結果
cv2.imwrite('data/beach_lantern_overlay.jpg', background)
print("已儲存為 data/beach_lantern_overlay.jpg")
