import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Model(nn.Module):
    def __init__(self, n_classes=5, width=16, dropout=0.1, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, width),  # 32 -> 16
            _block(width, width * 2),  # 16 -> 8
            _block(width * 2, width * 4),  # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, n_classes)
        )

    def forward(self, image):
        return {"logits": self.head(self.features(image))}
