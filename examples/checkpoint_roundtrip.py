import tempfile
from pathlib import Path

from speechloom.model import SpeechConfig, SpeechModel, load_checkpoint, save_checkpoint

model = SpeechModel(SpeechConfig(audio_bins=8, d_model=16, heads=2))
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'model.pt'
    save_checkpoint(path, model, step=0)
    restored, step = load_checkpoint(path)
assert restored.config == model.config
print({'step': step, 'config': restored.config})
