# 目的
使用 mediapipe 進行手勢辨識，並將結果顯示到網頁上。

# 需求
1. 使用 OpenCV 開啟攝影機，並且使用 mediapipe 進行手勢辨識。參考 https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer

2. 在原始影像上畫出手掌地標，並使用線條連結地標，完成後透過 server side event 技術將畫面傳給前端網頁

3. 前端網頁顯示影像以及手勢辨識結果

# 驗證與測試
自行驗證與測試影像是否傳到前端網頁上

# 儲存
將專案儲存在 handgesture 資料夾中，不可修改其餘資料夾中的程式