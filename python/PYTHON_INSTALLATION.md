# Installing for Python
## Standard Installation
Run
```
pip install eiko
```
You now have Eiko installed and ready for use.

### Optional dependencies
You can install additional dependencies for Eiko.
* To run visualization examples: `pip install -e eiko[examples]`
* To install with JAX: `pip install -e eiko[jax]`

The example scripts must be downloaded from Github: [Eiko examples](/../examples/python).

## Installation from Source
To download Eiko's source from Github and install locally, run:
```
git clone https://github.com/sebftw/Eiko.git
cd Eiko
pip install -e .
```

This includes the examples code, MATLAB support, and more.

If you intend on running the examples, you should run
```
pip install -e ".[examples]"
```