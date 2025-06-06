import cv2
import numpy as np

# 讀取圖檔
image = cv2.imread('data/zebra.jpg')

# 檢查圖檔是否成功讀取
if image is None:
    print("Error: Unable to load image.")
else:
    # 定義一個函數來將白色部分改為黑色
    def replace_white_with_black(image):
        black_color = (0, 0, 0)  # 黑色
        mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))  # 找到白色部分
        image[mask == 255] = black_color  # 將白色部分改為黑色
        return image

    # 定義第一次捲積的 3x3 矩形捲積核
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    convolved_image1 = cv2.filter2D(image, -1, kernel1)
    convolved_image1 = replace_white_with_black(convolved_image1)

    # 定義第二次捲積的 3x3 矩形捲積核
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    convolved_image2 = cv2.filter2D(convolved_image1, -1, kernel2)
    convolved_image2 = replace_white_with_black(convolved_image2)

    # 更新圖片拼接
    image = cv2.hconcat([image, convolved_image1, convolved_image2])

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
