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


@pytest.mark.model
def test_checkpoint_roundtrip():
    import tempfile
    from pathlib import Path

    import torch

    from speechloom.model import SpeechConfig, SpeechModel, load_checkpoint, save_checkpoint

    model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'model.pt'
        save_checkpoint(p, model, step=7)
        restored, step = load_checkpoint(p)
    assert step == 7
    assert restored.config == model.config
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )


@pytest.mark.model
def test_joint_loss_ignores_padding():
    import torch

    from speechloom.model import joint_loss

    logits = torch.zeros(1, 3, 268)
    labels = torch.tensor([[101, 260, -100]])
    loss = joint_loss(logits, labels, text_weight=1, audio_weight=2)
    assert float(loss) == pytest.approx(np.log(268))


@pytest.mark.model
def test_all_ignored_loss_rejected():
    import torch

    from speechloom.model import joint_loss

    with pytest.raises(ValueError, match='no supervised'):
        joint_loss(torch.zeros(1, 2, 268), torch.tensor([[-100, -100]]))


@pytest.mark.model
def test_greedy_eos_stops_and_restores_training_mode():
    import torch

    from speechloom.model import SpeechConfig, SpeechModel, generate

    model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2))
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[2] = 100
    result = generate(model, torch.zeros(1, 10), torch.tensor([10]), [1], 8)
    assert result == [1, 2]
    assert model.training
