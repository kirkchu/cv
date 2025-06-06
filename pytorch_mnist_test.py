import torch
from pytorch_mnist_model import LeNet as NeuralNetwork
import cv2
import matplotlib.pyplot as plt

ROOT = 'model'
TESTING = 'data/digit1.png'
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(f"{ROOT}/MNIST/model.pth", weights_only=True))

model.eval()

# 讀取圖片（灰階模式）
img = cv2.imread(TESTING, cv2.IMREAD_GRAYSCALE)

# 反相二值化（閾值200）
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# 尋找輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 將灰階圖轉為BGR以便畫彩色線
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 畫出所有輪廓及其 bounding box
for i, cnt in enumerate(contours):
    # 畫輪廓
    cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 2)
    # 計算並畫出 bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 裁切出每個輪廓範圍內的圖片
    roi = binary[y:y+h, x:x+w]
    #周圍填充 60 點的黑色邊框
    roi = cv2.copyMakeBorder(roi, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)
    roi = cv2.resize(roi, (28, 28))
    
    # 膨脹
    # roi = cv2.dilate(roi, np.ones((1, 1), np.uint8), iterations=1)
    # 侵蝕
    # roi = cv2.erode(roi, np.ones((2, 2), np.uint8), iterations=1)

    # 標準化到[0,1]並轉為tensor
    img_tensor = torch.tensor(roi, dtype=torch.float32) / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0, pred].item()
        print(f"預測結果: {pred}，信心值: {confidence:.4f}")
        # 在 bounding box 右下角標記預測結果與信心值（紅色字）
        label = f"{pred} ({confidence:.2f})"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = x + w
        text_y = y + h
        # 避免超出圖片邊界
        text_x = min(text_x, img_color.shape[1] - text_size[0])
        text_y = min(text_y, img_color.shape[0] - 5)
        cv2.putText(
            img_color, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

# 顯示圖片（使用 matplotlib）
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Contours")
plt.show()
