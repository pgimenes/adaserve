# AdaServe

## Getting Started

1. Follow the instructions [here](https://deepwok.github.io/mase/modules/documentation/getting_started.html) to set up a Mase environment using Conda or Docker.

2. Run the init.sh script to install the nightly versions of PyTorch and Transformers and initialize the mase submodule.

```bash
source scripts/init.sh
```

3. Verify the setup by running the CLI.

```bash
ada -h
```

4. Run one of the example scripts.

```bash
source examples/autosharding.sh
```