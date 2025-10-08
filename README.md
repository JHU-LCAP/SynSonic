# SynSonic: Augmenting Sound Event Detection through Text-to-Audio Diffusion ControlNet and Effective Sample Filtering

[![arXiv](https://img.shields.io/badge/arXiv-2409.10819-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.18603)

🟣 SynSonic is a framework that uses Text-to-Audio ControlNet for synthetic strongly-labeled audio data generation, improving the performance of sound event detection models.

## Steps

**1. Generate single-event audio clips using T2A ControlNet**  
- Install [EzAudio-ControlNet](https://github.com/haidog-yaqub/EzAudio)  
- Prepare trimmed reference audio samples  
- Generate audio variants using EzAudio-ControlNet:  

```python
from api.ezaudio import EzAudio_ControlNet
import torch
import soundfile as sf

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
controlnet = EzAudio_ControlNet(model_name='energy', device=device)

prompt = 'dog barking'
audio_path = 'egs/reference.mp3'  # Path to reference audio

sr, audio = controlnet.generate_audio(prompt, audio_path=audio_path)
sf.write(f"{prompt}_control.wav", audio, samplerate=sr)
```

**2. Filter the generated audio clips

**3. Synthesize strongly labeled audio mixtures
