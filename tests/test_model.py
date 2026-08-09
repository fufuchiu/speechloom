import numpy as np
import pytest


@pytest.mark.model
def test_joint_training_reaches_audio_encoder():
    import torch

    from speechloom.model import SpeechConfig, SpeechModel, train_step

    torch.manual_seed(4)
    torch.set_num_threads(1)
    model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2))
    audio = torch.randn(2, 24)
    lengths = torch.tensor([24, 18])
    inputs = torch.tensor([[1, 101, 3, 260], [1, 102, 3, 261]])
    labels = torch.tensor([[101, 3, 260, 2], [102, 3, 261, 2]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = [train_step(model, optimizer, audio, lengths, inputs, labels) for _ in range(20)]
    assert losses[-1] < losses[0] * 0.3
    assert model.frontend.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0


@pytest.mark.model
def test_future_tokens_cannot_change_past_outputs():
    import torch

    from speechloom.model import SpeechConfig, SpeechModel

    torch.manual_seed(3)
    model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2)).eval()
    audio = torch.randn(1, 16)
    lengths = torch.tensor([16])
    a = model(audio, lengths, torch.tensor([[1, 101, 102, 103]]))
    c = model(audio, lengths, torch.tensor([[1, 101, 120, 121]]))
    assert torch.allclose(a[:, :2], c[:, :2], atol=1e-6)


@pytest.mark.model
def test_padded_audio_is_invisible():
    import torch

    from speechloom.model import SpeechConfig, SpeechModel

    torch.manual_seed(3)
    model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2)).eval()
    audio = torch.randn(2, 18)
    changed = audio.clone()
    changed[0, 11:] = 1000
    lengths = torch.tensor([11, 18])
    ids = torch.tensor([[1, 101], [1, 102]])
    assert torch.allclose(model(audio, lengths, ids)[0], model(changed, lengths, ids)[0], atol=1e-6)
