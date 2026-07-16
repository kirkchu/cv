# 目的
撰寫一套人跟電腦玩剪刀、石頭、布的遊戲

# 規格需求
1. 使用 MediaPipe 的手勢辨識，參考 https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer。
2. 需畫出手掌地標，並且使用線條連接各地標。
3. 當使用者以 `石頭` 手勢前後搖晃時，表示準備動作，往前定住不動表示出拳完畢，電腦隨即出拳並且判定輸贏。

# 測試
使用 test.mov 作為使用者出拳的測試影片，檢查程式碼與邏輯是否有錯，如果有錯自行修正。

# 其他
1. 完成的程式碼儲存到 rps 資料夾中
2. 電腦出拳的圖案使用 rps 資料夾中的 paper.png、rock.png 與 scissors.png
3. **不可** 讀寫 rps 資料夾以外的檔案
4. 已使用 uv 建立虛擬環境，已安裝最新版 OpenCV 與 MediaPipe 函數庫，如需要安裝額外軟體，**必須** 取得使用者同意