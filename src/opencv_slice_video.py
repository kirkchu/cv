import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval_sec=5):
    # 取得影片檔名（不含副檔名）
    basename = os.path.splitext(os.path.basename(video_path))[0]
    # 建立輸出資料夾（若不存在則自動建立）
    os.makedirs(output_dir, exist_ok=True)
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_interval = int(fps * interval_sec)
    frame_idx = 0
    save_idx = 1
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{basename}_{save_idx:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        save_idx += 1
        frame_idx += frame_interval
    cap.release()
    print(f"完成擷取，總共儲存 {save_idx-1} 張圖片。")

# 範例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每隔指定秒數擷取影片frame並存檔")
    parser.add_argument("-i", "--input", dest="video_path", required=True, help="影片檔案路徑")
    parser.add_argument("-o", "--output", dest="output_dir", required=True, help="frame輸出資料夾")
    parser.add_argument("-s", "--seconds", dest="interval_sec", type=int, required=True, help="擷取間隔秒數")
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.interval_sec)
