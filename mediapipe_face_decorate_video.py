import cv2
import mediapipe as mp
import numpy as np

def overlay(original_image, overlay_image):
    overlay_image_rgb = overlay_image[:,:,:3]
    overlay_image_alpha = overlay_image[:,:,3:]

    overlay_mask = cv2.threshold(overlay_image_alpha, 0, 255, cv2.THRESH_BINARY)[1]
    original_mask = cv2.bitwise_not(overlay_mask)

    fig1 = cv2.bitwise_and(original_image, original_image, mask=original_mask)
    fig2 = cv2.bitwise_and(overlay_image_rgb, overlay_image_rgb, mask=overlay_mask)
    return cv2.add(fig1, fig2)

# 初始化 Mediapipe 的 Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 改為攝影機影像來源
cap = cv2.VideoCapture(0)  # 0 代表預設攝影機

# 設定解析度為 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 讀取裝飾圖檔
decorate_image_path = "face/crown.png"  # 替換為你的圖檔路徑
decorate_image = cv2.imread(decorate_image_path, cv2.IMREAD_UNCHANGED)
if decorate_image is None:
    raise FileNotFoundError(f"Image not found at {decorate_image_path}")

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)  # 水平翻轉影像
    # 將 BGR 圖片轉換為 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 偵測人臉地標
    results = face_mesh.process(rgb_image)
    # print(len(results.multi_face_landmarks))

    # 繪製地標
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
                # cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

                if idx == 54:
                    x_54 = x
                    y_54 = y
                if idx == 284:
                    x_284 = x
                    y_284 = y
                if idx == 10:
                    x_10 = x
                    y_10 = y

            # 繪製最外圍的白色線條
            hull = cv2.convexHull(np.array(points))
            cv2.polylines(image, [hull], isClosed=True, color=(255, 255, 255), thickness=2)

            # 計算 x_54, y_54 到 x_284, y_284 的斜率與角度
            dx = x_284 - x_54
            dy = y_284 - y_54
            angle = np.degrees(np.arctan2(dy, dx))

            # 繪製裝飾圖片（透明 PNG 合成）
            ratio = decorate_image.shape[0] / decorate_image.shape[1]
            decorate_image_resized = cv2.resize(
                decorate_image, 
                (int(x_284 - x_54), int((x_284 - x_54) * ratio)), 
                interpolation=cv2.INTER_AREA
            )

            # 旋轉裝飾圖（錨點設為中心點）
            (h_dec, w_dec) = decorate_image_resized.shape[:2]
            center = (w_dec // 2, h_dec // 2)  # 圖片中心點
            rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
            decorate_image_rotated = cv2.warpAffine(
                decorate_image_resized, rot_mat, (w_dec, h_dec), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
            )

            # 計算下緣中心點旋轉後的 x 軸偏移量
            bottom_center = np.array([[w_dec // 2, h_dec - 1]], dtype=np.float32)  # shape (1,2)
            bottom_center_rotated = cv2.transform(np.array([bottom_center]), rot_mat)[0][0]
            x_offset = int(bottom_center_rotated[0] - (w_dec // 2))

            # 以 x_10 為中心對齊裝飾圖，並修正 x 軸偏移
            x_center = x_10 - x_offset
            x1 = x_center - w_dec // 2
            x2 = x_center + (w_dec - w_dec // 2)
            y2 = y_10
            y1 = y2 - h_dec  # bottom 對齊 y_10

            # 若 top 超出貼圖區域，僅貼可見部分
            img_h, img_w = image.shape[:2]
            roi_x1 = max(0, x1)
            roi_x2 = min(img_w, x2)
            roi_y1 = max(0, y1)
            roi_y2 = min(img_h, y2)

            # 計算裝飾圖對應的裁切範圍
            crop_x1 = 0 if x1 >= 0 else -x1
            crop_x2 = decorate_image_rotated.shape[1] - (x2 - img_w) if x2 > img_w else decorate_image_rotated.shape[1]
            crop_y1 = 0 if y1 >= 0 else -y1
            crop_y2 = decorate_image_rotated.shape[0] - (y2 - img_h) if y2 > img_h else decorate_image_rotated.shape[0]

            # 檢查 ROI 與 crop 是否有效
            h_roi = roi_y2 - roi_y1
            w_roi = roi_x2 - roi_x1
            h_crop = crop_y2 - crop_y1
            w_crop = crop_x2 - crop_x1

            if h_roi > 0 and w_roi > 0 and h_crop > 0 and w_crop > 0:
                roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
                decorate_cropped = decorate_image_rotated[crop_y1:crop_y2, crop_x1:crop_x2]

                try:
                    if decorate_cropped.shape[0] != h_roi or decorate_cropped.shape[1] != w_roi:
                        decorate_cropped = cv2.resize(decorate_cropped, (w_roi, h_roi), interpolation=cv2.INTER_AREA)

                    # 若裝飾圖有 alpha 通道，進行 overlay
                    if decorate_cropped.shape[2] == 4 and roi.shape[0] > 0 and roi.shape[1] > 0:
                        merged = overlay(roi, decorate_cropped)
                        image[roi_y1:roi_y2, roi_x1:roi_x2] = merged
                    elif roi.shape[0] > 0 and roi.shape[1] > 0:
                        image[roi_y1:roi_y2, roi_x1:roi_x2] = decorate_cropped
                except Exception as e:
                    pass

    # 顯示結果
    cv2.imshow("Face Landmarks", image)
    if cv2.waitKey(1) == 27:  # 按 ESC 離開
        break

cap.release()
cv2.destroyAllWindows()
