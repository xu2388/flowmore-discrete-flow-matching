# FlowMore

<div align="center">
  <img src="assets/flowmore_animation.gif" width="500">
</div>

FlowMore is a minimal implementation of discrete flow matching for text generation. This is an adaption of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) to use flow matching. FlowMore takes name.txt containing a list of names, and learns to generate new names.

Flowmore is trained using a [discrete flow matching framework](https://arxiv.org/pdf/2407.15595) and utilizes a diffusion transformer (DiT) as the backbone. 

## Usage
```bash
pip install torch einops
python flowmore.py
```
