import cv2

# 讀取圖片
img = cv2.imread('data/zebra.jpg')

# 檢查圖片是否成功讀取
if img is None:
    print("無法讀取圖片")
else:
    # 取得圖片尺寸
    height, width = img.shape[:2]
    rect_size = 20
    spacing = 50
    color = (0, 0, 255)  # BGR 紅色
    thickness = 2

    rects = []
    img_no_rect = img.copy()
    y = 0
    while y + rect_size <= height:
        x = 0
        while x + rect_size <= width:
            cv2.rectangle(img, (x, y), (x + rect_size, y + rect_size), color, thickness)
            # 擷取矩形區域（從未畫過矩形的副本）
            rect = img_no_rect[y:y+rect_size, x:x+rect_size].copy()
            rects.append(rect)
            x += spacing
        y += spacing

    # 顯示原圖
    cv2.imshow('Zebra', img)

    # 拼接所有矩形成一張新圖（每8個一行）
    if rects:
        rows = []
        for i in range(0, len(rects), 8):
            row = cv2.hconcat(rects[i:i+8])
            rows.append(row)
        new_img = cv2.vconcat(rows)
        cv2.imshow('Rects', new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
