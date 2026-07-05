import torch.nn as nn

class LeNet (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_relu_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # in_channels=1, out_channels=6
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), 
        )
        self.conv2_relu_stack = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),  # in_channels=6, out_channels=16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*5*5, 120),  # 16個channel, 5x5 feature map
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1_relu_stack(x)
        x = self.conv2_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x