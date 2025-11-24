import torch
import torch.nn as nn
from torchvision import models

class ViolenceDetector(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=32):
        super(ViolenceDetector, self).__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.cnn_backbone = mobilenet.features
        self.cnn_out_features = 1280 
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features, 
            hidden_size=lstm_hidden_size, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.cnn_backbone(c_in)
        c_out = nn.functional.adaptive_avg_pool2d(c_out, (1, 1)).view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(c_out)
        return self.fc(lstm_out[:, -1, :])