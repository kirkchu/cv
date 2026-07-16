# CV 專案

✨ **電腦視覺 (Computer Vision) 範例集合**

本專案收錄多個電腦視覺示範與工具範例，涵蓋人臉辨識、手勢辨識、物件偵測/分類/分割、影像處理、OCR 等主題。
使用 **uv** 管理 Python 套件，模型統一放在根目錄 `model/` 下。

---

## 📁 專案結構

```
.
├── README.md
├── clsdataset/          # YOLO 分類訓練資料集（10 類動物）
│   ├── train/           #   訓練集
│   ├── val/             #   驗證集
│   └── test/            #   測試集
├── data/                # 範例圖檔/影片
├── face/                # 人臉資料庫與測試圖
│   ├── database/        #   註冊人臉（eun-bin, iu, tomcruise）
│   └── *.jpg            #   測試用臉部圖像
├── model/               # 模型檔案（.pt, .task）
│   ├── yolo26n.pt       #   YOLO 偵測
│   ├── yolo26n-cls.pt   #   YOLO 分類
│   ├── yolo26n-seg.pt   #   YOLO 分割
│   ├── yolo26n-pose.pt  #   YOLO 姿態
│   ├── sam3.pt          #   SAM 3 分割模型
│   ├── hand_landmarker.task
│   ├── face_landmarker.task
│   ├── gesture_recognizer.task
│   └── watch.pt         #   手錶偵測模型
└── src/                 # 所有 Python 腳本
    ├── deepface_*.py    #   人臉辨識（DeepFace）
    ├── mediapipe_*.py   #   MediaPipe 偵測
    ├── yolo_*.py        #   YOLO 系列
    ├── opencv_*.py      #   OpenCV 工具
    ├── sam3_*.py        #   SAM 3 分割
    ├── pytorch_mnist_*  #   MNIST 訓練/測試
    ├── colorfilters/    #   色彩濾鏡套件
    ├── handgesture/     #   手勢辨識 SSE 伺服器
    ├── rps/             #   剪刀石頭布遊戲
    └── web/             #   Web 串流伺服器
```

---

## 🔧 主要功能一覽

### 🧑 人臉辨識（DeepFace）

| 腳本 | 說明 |
|------|------|
| `deepface_find.py` | 從 `face/database` 搜尋相似人臉 |
| `deepface_find2.py` | 另一種搜尋實作（支援多張候選） |
| `deepface_verify.py` | 比對兩張圖像是否為同一人 |
| `deepface_embedding.py` | 產生人臉特徵向量（embedding） |
| `deepface_embedding_faiss.py` | 用 FAISS 建立索引並搜尋 |
| `deepface_embedding_pgvector.py` | 用 pgvector 儲存 embedding 並查詢 |
| `deepface_face_bbox.py` | 繪製人臉邊界框 |
| `deepface_test.py` | DeepFace 測試工具 |
| `deepface_video.py` | 攝影機即時人臉辨識 |

人臉資料庫位於 `face/database/`（目前包含：eun-bin、iu、tomcruise）。

### 🖐️ MediaPipe

| 腳本 | 說明 |
|------|------|
| `mediapipe_hand.py` | 手部 21 點地標即時偵測（模型：`model/hand_landmarker.task`） |
| `mediapipe_face.py` | 臉部 478 點地標偵測（模型：`model/face_landmarker.task`） |
| `mediapipe_face_livestream.py` | 攝影機即時臉部地標 |
| `mediapipe_face_decorate.py` | 臉部地標 + 合成裝飾圖（如皇冠 `face/crown.png`） |
| `mediapipe_face_decorate_video.py` | 影片版裝飾（支援攝影機或影片檔） |

### 🤖 YOLO（Ultralytics）

| 腳本 | 說明 |
|------|------|
| `yolo_imageclassification.py` | 影像分類（使用 `yolo26n-cls.pt`） |
| `yolo_cls_train.py` | 使用 `clsdataset/` 訓練分類模型 |
| `yolo_cls_test.py` | 測試分類模型 |
| `yolo_objectdetection_manual.py` | 物件偵測（`yolo26n.pt`） |
| `yolo_objectdetection_sv.py` | 使用 Supervision 繪製偵測結果 |
| `yolo_objectdetection_video_sv.py` | 影片物件偵測 + Supervision 繪圖 |
| `yolo_pose.py` | 姿態估計（`yolo26n-pose.pt`） |
| `yolo_traffic.py` | 交通場景偵測 |
| `yolo_traffic_density.py` | 車流密度分析 |
| `yolo_create_trainingset.py` | 建立自訂訓練資料集 |
| `yolo_video2jpg.py` | 影片轉圖片（供標註使用） |
| `yolo_ndjson.py` | NDJSON 格式處理 |
| `yolo_colab_ndjson.ipynb` | Colab 筆記本：NDJSON 標註 |
| `yolo_colab_objectdetection.ipynb` | Colab 筆記本：物件偵測 |

### 🎨 SAM 3（Segment Anything Model 3）

| 腳本 | 說明 |
|------|------|
| `sam3_chair.py` | 文字提示 "chair" 分割椅子並繪製遮罩 |
| `sam3_chat.py` | 互動式對話分割（問答式指定目標） |
| `sam3_video.py` | 攝影機即時語意分割 |

需要 `ultralytics>=8.3.237`，模型：`model/sam3.pt`（~3.4GB）。

### 🛠️ OpenCV 工具集

| 腳本 | 說明 |
|------|------|
| `opencv_image.py` | 基本影像讀寫顯示 |
| `opencv_video.py` | 基本影片讀取播放 |
| `opencv_binary.py` | 二值化處理 |
| `opencv_canny.py` | Canny 邊緣偵測 |
| `opencv_erode_dilate.py` | 侵蝕與膨脹 |
| `opencv_hull.py` | 凸包計算 |
| `opencv_hull_draw.py` | 凸包繪製 |
| `opencv_findcnt.py` | 輪廓尋找 |
| `opencv_findcnt_digit.py` | 數字輪廓辨識 |
| `opencv_features.py` | 特徵點偵測 |
| `opencv_find_features_orb.py` | ORB 特徵尋找 |
| `opencv_find_features_cnn.py` | 用 CNN 模型尋找特徵 |
| `opencv_match_feature_orb.py` | ORB 特徵匹配 |
| `opencv_color_detection.py` | 顏色偵測（HSV 範圍） |
| `opencv_get_hsv_value.py` | 取得 HSV 數值工具 |
| `opencv_roi.py` | 感興趣區域（ROI）選取 |
| `opencv_overlay.py` | 影像疊加 |
| `opencv_pooling.py` | 影像池化（Max/Avg Pooling） |
| `opencv_background_knn.py` | KNN 背景去除 |
| `opencv_background_substractor.py` | 背景減除 |
| `opencv_tracker.py` | 物件追蹤 |

### 🎮 手勢辨識與遊戲

| 腳本/目錄 | 說明 |
|-----------|------|
| `handgesture/server.py` | MediaPipe GestureRecognizer 手勢辨識 SSE 串流伺服器，附前端 `index.html` |
| `rps/rps_game.py` | 剪刀石頭布遊戲（攝影機或影片），使用 `model/gesture_recognizer.task` |

### 🌐 Web 串流伺服器

| 腳本 | 說明 |
|------|------|
| `web/server.py` | 內建 HTTP 伺服器 + 攝影機串流（CGI 模式），附前端 `index.html` |

### 🎨 色彩濾鏡套件

| 套件 | 說明 |
|------|------|
| `colorfilters/` | Python 套件，提供多種色彩濾鏡（HSV 範圍遮罩、顏色混合等），可透過 `python -m colorfilters` 執行 |

### 🔢 其他

| 腳本 | 說明 |
|------|------|
| `ocr_carplate.py` | 使用 EasyOCR 的車牌辨識（範例圖：`data/carplate.jpg`） |
| `pytorch_mnist_train.py` | PyTorch MNIST 訓練 |
| `pytorch_mnist_test.py` | MNIST 模型測試 |
| `pytorch_mnist_model.py` | MNIST 模型定義 |

---

## 📦 分類資料集（clsdataset）

YOLO 分類訓練用的資料集，包含 10 類動物，已分割為 train / val / test：

```
clsdataset/
├── train/   (10 類)
├── val/     (10 類)
└── test/    (10 類)
```

類別：cat, dog, elephant, fox, giraffe, lion, monkey, owl, tiger, zebra

搭配 `yolo_cls_train.py` 使用 `yolo26n-cls.pt` 進行遷移學習訓練。

---

## 🚀 快速上手

### 前置需求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（推薦套件管理工具）

### 安裝

```bash
# 初始化虛擬環境並安裝套件
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt   # 或直接 uv pip install 所需套件
```

### 模型檔案

模型檔案已放在根目錄 `model/` 下，腳本透過 `src/../model/<file>` 引用：

| 模型 | 用途 | 來源 |
|------|------|------|
| `yolo26n.pt` | YOLO 偵測 | Ultralytics |
| `yolo26n-cls.pt` | YOLO 分類 | Ultralytics |
| `yolo26n-seg.pt` | YOLO 分割 | Ultralytics |
| `yolo26n-pose.pt` | YOLO 姿態 | Ultralytics |
| `sam3.pt` (~3.4GB) | SAM 3 分割 | Ultralytics |
| `hand_landmarker.task` | 手部地標 | MediaPipe |
| `face_landmarker.task` | 臉部地標 | MediaPipe |
| `gesture_recognizer.task` | 手勢辨識 | MediaPipe |
| `watch.pt` | 手錶偵測 | 自訂訓練 |

### 執行範例

```bash
# 人臉搜尋
python src/deepface_find.py

# 人臉驗證
python src/deepface_verify.py

# 手部地標偵測
python src/mediapipe_hand.py

# 臉部地標 + 皇冠裝飾
python src/mediapipe_face_decorate.py

# YOLO 物件偵測
python src/yolo_objectdetection_manual.py

# YOLO 分類訓練
python src/yolo_cls_train.py

# SAM 3 文字提示分割
python src/sam3_chair.py

# 剪刀石頭布遊戲
python src/rps/rps_game.py

# 手勢辨識 SSE 伺服器（開啟後瀏覽器連線 http://localhost:8000）
python src/handgesture/server.py

# Web 串流伺服器（開啟後瀏覽器連線 http://localhost:8000）
python src/web/server.py

# 色彩濾鏡
python -m src.colorfilters

# 車牌辨識
python src/ocr_carplate.py
```

---

## ⚠️ 已知問題

### opencv-python 可能缺少 haarcascade XML

`opencv-python` 的 PyPI wheel 不一定會打包 OpenCV 原始碼中的 Cascade 分類器 XML 檔，因此 `cv2/data/` 目錄下可能找不到 `haarcascade_frontalface_default.xml` 等檔案，導致 DeepFace 的 OpenCV face detector 拋出 `ValueError`。

解決方式：從 OpenCV GitHub 手動下載到 `cv2/data/` 目錄：

```bash
curl -L -o /path/to/venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml \
  "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

curl -L -o /path/to/venv/lib/python3.12/site-packages/cv2/data/haarcascade_eye.xml \
  "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
```

---

## 📝 注意事項

- 本專案使用 **uv** 管理套件，`.venv` 內無 pip，需用 `uv pip ...` 安裝套件。
- 多數腳本會使用 `data/` 或 `face/` 底下的圖檔，若移動資源請同步修改腳本中的路徑。
- 模型檔統一放在根目錄 `model/`，非 `src/` 內。
- `sam3.pt` 約 3.4GB，若未下載會造成 SAM 3 相關腳本無法執行。
- 建議使用 **Git LFS** 管理大型二進位檔（如 `sam3.pt`、`*.jpg`、`*.mov` 等）以減少儲存庫大小。
