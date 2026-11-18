
import torch

from src.model import AudioCNN


def test_model_output_shape():
    model = AudioCNN(num_classes=4)

    # 2 аудіозаписи: 1 канал, 64 mel-частоти, 100 часових кроків
    x = torch.randn(2, 1, 64, 100)

    output = model(x)

    assert output.shape == (2, 4)


def test_model_output_is_tensor():
    model = AudioCNN(num_classes=4)
    x = torch.randn(1, 1, 64, 100)

    output = model(x)

    assert isinstance(output, torch.Tensor)


def test_model_has_correct_number_of_parameters():
    model = AudioCNN(num_classes=4)

    assert sum(p.numel() for p in model.parameters()) == 56004
