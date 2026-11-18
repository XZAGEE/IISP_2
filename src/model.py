
import torch.nn as nn


class AudioCNN(nn.Module):
    """CNN для класифікації аудіо за Mel-Spectrogram."""

    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()

        # Згорткові шари для виділення ознак
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Повнозв'язний шар для класифікації
        self.fc = nn.Linear(12800, num_classes)

    def forward(self, x):
        # Передача даних через згорткові шари
        x = self.conv(x)

        # Перетворення тензора на вектор ознак
        x = x.view(x.size(0), -1)

        # Отримання результатів класифікації
        x = self.fc(x)

        return x
