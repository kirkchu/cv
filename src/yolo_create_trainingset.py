import os
import shutil
import random
import glob

# Define the project directory
# 請將此路徑替換為您的專案目錄
project_dir = '/Users/ckk/venv/cv/project'

# Define source directories
source_images_dir = os.path.join(project_dir, 'images')
source_labels_dir = os.path.join(project_dir, 'labels')

# Define destination directories
train_dir = os.path.join(project_dir, 'train')
val_dir = os.path.join(project_dir, 'val')
test_dir = os.path.join(project_dir, 'test')

# Create destination directories if they don't exist
print(f"正在建立目標目錄: {train_dir}, {val_dir}, {test_dir}")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create images and labels subdirectories within destination directories
print("正在建立目標目錄內的 images 和 labels 子目錄...")
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Get list of image files and create a map from basename to full path
# Assuming common image extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files_map = {}
print(f"正在掃描原始圖片目錄: {source_images_dir}")
for ext in image_extensions:
    for f in glob.glob(os.path.join(source_images_dir, ext)):
        basename = os.path.splitext(os.path.basename(f))[0]
        image_files_map[basename] = f

# Get list of basenames from the image files found
basenames = list(image_files_map.keys())

# Ensure there are files to process
if not basenames:
    print(f"在 {source_images_dir} 中找不到任何圖片檔案。程式結束。")
else:
    # Shuffle the basenames
    print("正在隨機排序檔案列表...")
    random.shuffle(basenames)

    # Calculate split sizes
    total_files = len(basenames)
    train_size = int(total_files * 0.7)
    val_size = int(total_files * 0.2)
    test_size = total_files - train_size - val_size # Assign remaining to test

    # Split basenames
    train_basenames = basenames[:train_size]
    val_basenames = basenames[train_size:train_size + val_size]
    test_basenames = basenames[train_size + val_size:]

    print(f"總檔案對數: {total_files}")
    print(f"訓練集大小: {len(train_basenames)}")
    print(f"驗證集大小: {len(val_basenames)}")
    print(f"測試集大小: {len(test_basenames)}")

    # Function to move a file pair
    def move_file_pair(basename, target_dir):
        source_image_path = image_files_map.get(basename)
        # Assuming label files have .txt extension
        source_label_path = os.path.join(source_labels_dir, f"{basename}.txt")

        if source_image_path and os.path.exists(source_label_path):
            # Determine target paths
            target_image_dir = os.path.join(target_dir, 'images')
            target_label_dir = os.path.join(target_dir, 'labels')

            target_image_path = os.path.join(target_image_dir, os.path.basename(source_image_path))
            target_label_path = os.path.join(target_label_dir, os.path.basename(source_label_path))

            # Move files
            try:
                shutil.move(source_image_path, target_image_path)
                shutil.move(source_label_path, target_label_path)
                # print(f"Moved {basename} to {target_dir}")
            except FileNotFoundError:
                 print(f"警告: 找不到 {basename} 的檔案對。跳過。")
            except Exception as e:
                 print(f"移動 {basename} 時發生錯誤: {e}")

        elif not source_image_path:
             print(f"警告: 找不到 basename 為 {basename} 的圖片檔案。跳過。")
        elif not os.path.exists(source_label_path):
             print(f"警告: 找不到 basename 為 {basename} 的標籤檔案。跳過。")


    # Move files for each split
    print("正在移動訓練集檔案...")
    for basename in train_basenames:
        move_file_pair(basename, train_dir)

    print("正在移動驗證集檔案...")
    for basename in val_basenames:
        move_file_pair(basename, val_dir)

    print("正在移動測試集檔案...")
    for basename in test_basenames:
        move_file_pair(basename, test_dir)

    # Check and remove original directories if empty
    print("正在檢查原始目錄...")
    if os.path.exists(source_images_dir) and not os.listdir(source_images_dir):
        print(f"移除空的目錄: {source_images_dir}")
        os.rmdir(source_images_dir)
    else:
        print(f"{source_images_dir} 不是空的或不存在。")

    if os.path.exists(source_labels_dir) and not os.listdir(source_labels_dir):
        print(f"移除空的目錄: {source_labels_dir}")
        os.rmdir(source_labels_dir)
    else:
        print(f"{source_labels_dir} 不是空的或不存在。")

    print("處理完成。")
