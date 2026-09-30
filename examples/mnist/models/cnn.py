import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes=10, width=32, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 2 * 7 * 7, n_classes)
        )

    def forward(self, image):
        return {"logits": self.head(self.features(image))}
