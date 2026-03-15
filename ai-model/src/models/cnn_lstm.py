import torch
import torch.nn as nn
from torchvision import models

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(256, 2)  # 2 classes

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])
