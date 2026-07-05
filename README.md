# CV 專案（src）

✨ **專案概述**

本專案收錄多個放在 `src/` 底下的電腦視覺示範與工具範例，包含人臉辨識（DeepFace）、MediaPipe 臉部/手部地標偵測與裝飾、基於 YOLO 的偵測/分類範例，以及各式 OpenCV 工具腳本。

---

## 🔧 主要功能一覽

- **人臉 / DeepFace**
  - `src/deepface_find.py`, `src/deepface_find2.py` — 從 `src/face/database` 搜尋相似人臉。
  - `src/deepface_verify.py` — 使用 DeepFace 比對兩張圖像。
  - `src/deepface_embedding.py`, `src/deepface_embedding_faiss.py`, `src/deepface_test.py` — 產生 embedding 並用 FAISS 做最近鄰搜尋。
  - `src/deepface_video.py` — 使用攝影機即時辨識人臉。
  - 人臉資料位於 `src/face/database/*`。

- **MediaPipe**
  - `src/mediapipe_hand.py` — MediaPipe 手部地標偵測（模型路徑：`src/../model/hand_landmarker.task`）。
  - `src/mediapipe_face.py`, `src/mediapipe_face_livestream.py` — 臉部地標範例（模型：`src/../model/face_landmarker.task`）。
  - `src/mediapipe_face_decorate.py`, `src/mediapipe_face_decorate_video.py` — 使用偵測到的地標合成裝飾圖（例如皇冠）。

- **YOLO（Ultralytics）**
  - `src/yolo_imageclassification.py`, `src/yolo_objectdetection_manual.py`, `src/yolo_objectdetection_sv.py`, `src/yolo_pose.py`, `src/yolo_traffic.py`, `src/yolo_traffic_density.py` — 分類、偵測、姿態、分割等示例。
  - 模型路徑以 `src/../model/...` 為準（model 資料夾位於 `src/` 的上層）。

- **OpenCV 工具集**
  - `src/opencv_*` — 包含 Canny、輪廓、特徵配對、ORB/SIFT/CNN 範例、影像 pooling、影片工具等多個腳本。

- **其他**
  - `src/ocr_carplate.py` — 使用 EasyOCR 的車牌範例（範例圖：`src/data/carplate.jpg`）。
  - `src/pytorch_mnist_*` — MNIST 訓練與測試工具。
  ---

## ⚠️ 已知問題：opencv-python 可能缺少 haarcascade XML

`opencv-python` 的 PyPI wheel 不一定會打包 OpenCV 原始碼中的 Cascade 分類器 XML 檔（位於 OpenCV 官方 repo 的 `data/haarcascades/`），因此 `cv2/data/` 目錄下可能找不到 `haarcascade_frontalface_default.xml` 等檔案。

解決方式：從 OpenCV GitHub 手動下載需要的 XML 檔到 `cv2/data/` 目錄：

```bash
curl -L -o /path/to/venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml \
  "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

curl -L -o /path/to/venv/lib/python3.12/site-packages/cv2/data/haarcascade_eye.xml \
  "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
```

---

## ▶️ 快速上手

1. 建議使用 `uv` 初始化虛擬環境：

   ```bash
   uv init      # 初始化並建立虛擬環境（視 uv 工具而定會自動啟用）
   # 若需要手動啟用，請使用：uv activate
   pip install -r requirements.txt
   ```

2. 模型檔：請把模型放在專案根目錄的 `model/` 資料夾（在 `src/` 的上層），範例腳本會引用 `src/../model/<file>`。

3. 執行範例：

   - DeepFace 找相似範例：
     ```bash
     python src/deepface_find.py
     ```

   - MediaPipe 手部偵測：
     ```bash
     python src/mediapipe_hand.py
     ```

   - YOLO 偵測範例：
     ```bash
     python src/yolo_objectdetection_manual.py
     ```

4. 多數腳本會使用 `src/data/` 或 `src/face/` 底下的圖檔，若移動資源請同步修改腳本中的路徑。

---

## ⚠️ 注意事項與建議

- 本專案已將 `src/` 內容加入版本控制，包含大量影像檔。建議使用 **Git LFS** 管理大型二進位檔（例如：`git lfs track "src/**/*.jpg"`）以減少儲存庫大小。
- 模型檔應放在專案根目錄的 `model/`（非 `src/` 內）。若你已把 `model/` 移動，請確認其位於 `./model/`（相對於 repo root）。
- 若希望範例更具可移植性，可新增短小的 `scripts/` 包裝腳本，統一設定 model/data 的路徑。
