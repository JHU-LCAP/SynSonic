# SynSonic: Augmenting Sound Event Detection through Text-to-Audio Diffusion ControlNet and Effective Sample Filtering

[![arXiv](https://img.shields.io/badge/arXiv-2409.10819-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.18603)

🟣 SynSonic is a framework that uses Text-to-Audio ControlNet for synthetic strongly-labeled audio data generation, improving the performance of sound event detection models.

## Steps

**1. Generate single-event audio clips using T2A ControlNet**  
- Install [EzAudio-ControlNet](https://github.com/haidog-yaqub/EzAudio)  
- Prepare trimmed reference audio samples (Please follow soundbank used in [DCASE SED](https://project.inria.fr/desed/download/synthetic-data/))
- Generate audio variants using EzAudio-ControlNet:  

```python
from api.ezaudio import EzAudio_ControlNet
import torch
import soundfile as sf

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
controlnet = EzAudio_ControlNet(model_name='energy', device=device)

prompt = 'dog barking'
audio_path = 'real_dog_barking.wav'  # Path to reference audio

sr, audio = controlnet.generate_audio(prompt, audio_path=audio_path)
sf.write(f"{prompt}_gen.wav", audio, samplerate=sr)
```

**2. Filter the generated audio clips**  
- Set up [Dasheng](https://github.com/XiaoMi/dasheng) and [Laion-CLAP](https://huggingface.co/laion/clap-htsat-fused)  
- Use Dasheng to compute logits  
- Use CLAP to compute text–audio similarity
- Rank samples separately by logits and similarity  
- Re-rank samples using a weighted score:
  - score = w1 * r1 + w2 * r2, where r1 is the rank from logits and r2 is the rank from similarity
- Select the top \(k\%\) of samples based on the final score  


**3. Synthesize strongly labeled audio mixtures
- Please follow [DCASE SED](https://project.inria.fr/desed/download/synthetic-data/)
- Download background soundbank: [https://zenodo.org/records/6026841/files/DESED_synth_soundbank.tar.gz](https://zenodo.org/records/6026841/files/DESED_synth_soundbank.tar.gz)
- Follow this pipeline: [https://github.com/turpaultn/DESED/blob/master/desed/desed/generate_synthetic.py](https://github.com/turpaultn/DESED/blob/master/desed/desed/generate_synthetic.py) to create mixtures with strong labels.
