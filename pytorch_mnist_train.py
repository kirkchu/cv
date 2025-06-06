import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pytorch_mnist_model import LeNet as NeuralNetwork
import matplotlib.pyplot as plt

ROOT = 'model'
BATCH_SIZE = 800
LR = 0.01
EPOCH = 40 
# 1. 資料預處理與載入
train_dataset = datasets.MNIST(root=ROOT, train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root=ROOT, train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

# 2. 定義模型
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model) 

# 3. 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
test_losses = []
test_accuracies = []

# 4. 訓練模型
def train(epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

# 5. 測試模型
best = 0 
def test():
    global best
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    if accuracy >= best:
        best = accuracy
        torch.save(model.state_dict(), f"{ROOT}/MNIST/model.pth")
        print("======= Saved PyTorch Model State to model.pth")

for epoch in range(1, EPOCH + 1):
    train(epoch)
    test()

# torch.save(model.state_dict(), "data/MNIST/model.pth")
# print("Saved PyTorch Model State to model.pth")

# 畫圖
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()