# AdaServe

## Getting Started

1. Follow the instructions [here](https://deepwok.github.io/mase/modules/documentation/getting_started.html) to set up a Mase environment using Conda or Docker.

2. Install the nightly release of Pytorch.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

3. Run the init.sh script to initialize the mase submodule.

```bash
source init.sh
```

4. Verify the setup by running the manual sharding examples.

```
source examples/manual.py
```