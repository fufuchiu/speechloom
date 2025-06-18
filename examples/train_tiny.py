import torch

from speechloom.model import SpeechConfig, SpeechModel, train_step

torch.manual_seed(4)
torch.set_num_threads(1)
model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
audio = torch.randn(2, 32)
lengths = torch.tensor([32, 24])
inputs = torch.tensor([[1, 101, 3, 260], [1, 102, 3, 261]])
labels = torch.tensor([[101, 3, 260, 2], [102, 3, 261, 2]])
losses = [train_step(model, optimizer, audio, lengths, inputs, labels) for _ in range(25)]
print({'initial_loss': losses[0], 'final_loss': losses[-1]})
assert losses[-1] < losses[0]
