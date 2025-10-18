import torch
import torch.nn as nn
from model.attention import CBAM

class CNNWithCBAM(nn.Module):
    """
    CNN model with a CBAM attention block for counterfeit currency detection.
    """
    def __init__(self, num_classes=2):
        super(CNNWithCBAM, self).__init__()
        # Example backbone — you can adjust filter sizes/layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # reduce H/W by 2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Insert CBAM after layer3
        self.cbam = CBAM(in_planes=128, ratio=8, kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # To test the model architecture
    model = CNNWithCBAM(num_classes=2)
    print(model)
    # Create a dummy input tensor
    dummy_input = torch.randn(4, 3, 224, 224) # (batch_size, channels, height, width)
    # Get model output
    output = model(dummy_input)
    print("Output shape:", output.shape)

