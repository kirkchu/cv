import requests
import os
import random
from urllib.parse import urlparse, unquote

# 步驟 1 & 2: 將 curl 指令換成 python requests，並將 query 拉出來變成變數
# 請將 YOUR_API_KEY 替換為您的實際 Pexels API 金鑰
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.pexels.com/v1/search"

# 步驟 3: 將變數 query 改為陣列型態
queries = ['cat', 'dog', 'elephant']

# 設定每頁獲取的圖片數量，為了獲取足夠的圖片進行分割，這裡設定一個較大的值
PER_PAGE = 50 # 您可以根據需要調整此值

# 步驟 4: 建立 clsdataset/train, clsdataset/val, clsdataset/test 資料夾
OUTPUT_DIR = 'clsdataset'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

# 確保基礎輸出資料夾存在
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def download_and_save(photo_list, target_dir):
    """
    下載圖片列表中的圖片並儲存到目標資料夾。
    """
    for photo in photo_list:
        # 步驟 5: 參考 json 結構，取得 tiny 圖片的 url
        img_url = photo['src']['tiny']
        # 從 url 中提取檔名，例如 3573351.png
        parsed_url = urlparse(img_url)
        # 處理可能的查詢參數，只取路徑的最後一部分作為檔名
        filename = os.path.basename(parsed_url.path)
        # 處理 url 解碼，例如 %2F 轉換為 /
        filename = unquote(filename)
        # 確保檔名不包含無效字元，這裡簡單處理，只取數字和副檔名
        # 更嚴謹的做法可能需要正則表達式
        if '.' in filename:
             name_part, ext = filename.rsplit('.', 1)
             # 移除非數字字元，只保留數字作為主要檔名
             clean_name_part = ''.join(filter(str.isdigit, name_part))
             if clean_name_part:
                 filename = f"{clean_name_part}.{ext}"
             else:
                 # 如果沒有數字，則使用 photo['id'] 作為檔名
                 filename = f"{photo['id']}.{ext}"
        else:
             # 如果沒有副檔名，則使用 photo['id'] 作為檔名，並假設為 jpg
             filename = f"{photo['id']}.jpg" # 假設常見圖片格式

        filepath = os.path.join(target_dir, filename)

        # 檢查檔案是否已存在，避免重複下載
        if os.path.exists(filepath):
            print(f"檔案已存在，跳過下載: {filename}")
            continue

        try:
            img_data = requests.get(img_url, stream=True)
            img_data.raise_for_status() # 檢查 HTTP 請求是否成功

            with open(filepath, 'wb') as f:
                for chunk in img_data.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"下載並儲存成功: {filename} -> {filepath}")

        except requests.exceptions.RequestException as e:
            print(f"下載圖片失敗 {img_url}: {e}")
        except IOError as e:
            print(f"儲存圖片失敗 {filepath}: {e}")


# 步驟 3 (續): 使用迴圈依次呼叫 api
for query_item in queries:
    print(f"正在獲取 '{query_item}' 的圖片...")

    params = {
        "query": query_item,
        "per_page": PER_PAGE
    }
    headers = {
        "Authorization": API_KEY
    }

    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status() # 如果響應狀態碼不是 200，則拋出 HTTPError

        data = response.json()
        photos = data.get('photos', [])

        if not photos:
            print(f"找不到 '{query_item}' 的圖片或 API 響應中沒有 'photos' 鍵。")
            continue

        print(f"找到 {len(photos)} 張 '{query_item}' 的圖片。")

        # 步驟 5 (續): 根據 70%, 20%, 10% 的比例分割並儲存
        random.shuffle(photos) # 打亂圖片列表以實現隨機分割

        total_photos = len(photos)
        train_split = int(total_photos * 0.7)
        val_split = int(total_photos * 0.2)
        # 剩餘的圖片用於測試集
        test_split = total_photos - train_split - val_split

        train_photos = photos[:train_split]
        val_photos = photos[train_split : train_split + val_split]
        test_photos = photos[train_split + val_split :]

        # 為當前 query_item 建立子資料夾
        current_train_dir = os.path.join(TRAIN_DIR, query_item)
        current_val_dir = os.path.join(VAL_DIR, query_item)
        current_test_dir = os.path.join(TEST_DIR, query_item)

        os.makedirs(current_train_dir, exist_ok=True)
        os.makedirs(current_val_dir, exist_ok=True)
        os.makedirs(current_test_dir, exist_ok=True)

        print(f"下載並儲存 '{query_item}' 訓練集 ({len(train_photos)} 張)...")
        download_and_save(train_photos, current_train_dir)

        print(f"下載並儲存 '{query_item}' 驗證集 ({len(val_photos)} 張)...")
        download_and_save(val_photos, current_val_dir)

        print(f"下載並儲存 '{query_item}' 測試集 ({len(test_photos)} 張)...")
        download_and_save(test_photos, current_test_dir)

        print(f"'{query_item}' 的圖片下載和分割完成。")

    except requests.exceptions.RequestException as e:
        print(f"呼叫 Pexels API 失敗 (查詢: '{query_item}'): {e}")
    except Exception as e:
        print(f"處理 '{query_item}' 時發生錯誤: {e}")

print("所有查詢關鍵字的圖片處理完成。")
