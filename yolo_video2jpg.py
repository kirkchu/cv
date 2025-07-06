import cv2
import os

# Define video file path and output directory
video_path = '/Users/ckk/venv/cv/example/IMG_9466.MOV' # 請確認此路徑正確
output_dir = '/Users/ckk/venv/cv/example/rawdata'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_count = 0
saved_image_count = 0

print(f"Processing video: {video_path}")

while True:
    # Read a frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        break # End of video

    frame_count += 1

    # Save every 5th frame
    if frame_count % 5 == 0:
        saved_image_count += 1
        # Format the filename with leading zeros (e.g., 001.jpg)
        image_filename = os.path.join(output_dir, f'{saved_image_count:03d}.jpg')

        # Save the frame as a JPG file
        cv2.imwrite(image_filename, frame)
        print(f"Saved {image_filename} (Frame {frame_count})")

# Release the video capture object
cap.release()

print(f"Finished processing. Total frames processed: {frame_count}. Total images saved: {saved_image_count}")
