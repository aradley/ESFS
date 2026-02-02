# ESFS

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/aradley/ESFS/releases/tag/v1.0.0)

**Version 1.0.0** - Paper Release

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

> **Note:** This is the frozen release accompanying our publication. For the latest version with additional features (including Apple Silicon GPU support), see the [latest release](https://github.com/aradley/ESFS).

Please see our manuscript for details regarding ESFS -

Go to the Example_Workflows folder to see some example workflows that you may adapt for your own data.

Datasets for reproducing the example workflows may be found at the following Mendeley Data repository -

### Installation (Paper Version)

To install the exact version used in our paper:

```
pip install git+https://github.com/aradley/ESFS.git@v1.0.0
```

### Installation (Latest)
Either install this repository directly via:

```
pip install git+https://github.com/aradley/ESFS.git
```

or clone and then install:

```
git clone git@github.com:aradley/ESFS.git
cd ESFS
pip install .
```

You should do this within an environment, using something like `uv`, `venv`, or `conda`.

## Software overview

![ESFS is comprised of 3 main algorithms - ES-GSS, ES-CCF and ES-FMG](Figure_1.png)

## GPU acceleration

For large datasets, users may wish to use the GPU accelerated version of ESFS to perform ES correlation metric calculations.

To install the GPU enabled version of ESFS, please use the following which will icorperate CuPy into the installation:

```
pip install "esfs[gpu] @ git+https://github.com/aradley/ESFS.git"
```

or clone and then install:

```
git clone git@github.com:aradley/ESFS.git
cd ESFS
pip install '.[gpu]'
```


By default the GPU version of ESFS will be loaded when running ``` import esfs ``` if a compatible version of CUDA is installed on the machine, and a message will print saying that the GPU version is in use. If you have installed the GPU version but CUDA is not avaialble on your machine, ESFS will default to to the CPU version and print a message telling you it has done so.

If users wish to force ESFS to run using CPUs even when CUDA is available, they may do so by running ``` esfs.configure(gpu=False) ``` after running ``` import esfs ```.
