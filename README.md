# SynSonic: Augmenting Sound Event Detection through Text-to-Audio Diffusion ControlNet and Effective Sample Filtering

[![arXiv](https://img.shields.io/badge/arXiv-2409.10819-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.18603)

🟣 SynSonic is a framework that uses Text-to-Audio ControlNet for synthetic strongly-labeled audio data generation, improving the performance of sound event detection models.

## Steps

1. Generate single-event audio clips using T2A ControlNet
-  Install [EzAudio-ControlNet](https://github.com/haidog-yaqub/EzAudio)
-  Prepare reference audio samples (should be trimmed)
-  Use EzAudio-ControlNet to generate variants for reference audio samples:
```python

from api.ezaudio import EzAudio
import torch
import soundfile as sf

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
controlnet = EzAudio_ControlNet(model_name='energy', device=device)

prompt = 'dog barking'
# path for audio reference
audio_path = 'egs/reference.mp3'

sr, audio = controlnet.generate_audio(prompt, audio_path=audio_path)
sf.write(f"{prompt}_control.wav", audio, samplerate=sr)
```

3. Filter the generated audio clips

4. Synthesize strongly labeled audio mixtures
