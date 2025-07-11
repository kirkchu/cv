from deepface import DeepFace

# 設定來源圖片與資料集資料夾
source_img = "face/find_test1.jpg"
db_path = "face/database"

# 執行人臉辨識
results = DeepFace.find(img_path=source_img, db_path=db_path)

# 使用 Claude Sonnet 4 模型
# 將 pandas DataFrame 轉換為 Python 列表
results_list = results[0].to_dict('records') if len(results) > 0 else []

# 輸出結果
print(results_list)

