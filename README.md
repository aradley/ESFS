# ESFS

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

Please see our manuscript for details regarding ESFS - 

Go to the Example_Workflows folder to see some example workflows that you will hopefully be able to easily adapt for your own data.

Datasets for reproducing the example workflows may be found at the following Mendeley Data repsitory - 

### Installation
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

#### GPU Installation

To install ESFS with CuPy so that it can run on a GPU, please modify the above lines accordingly:

```
pip install "esfs[gpu] @ git+https://github.com/aradley/ESFS.git"
```
or
```
pip install '.[gpu]'
```

## Software overview

![ESFS is comprised of 3 main algorithms - ES-GSS, ES-CCF and ES-FMG](Figure_1.png)

## GPU acceleration

For large datasets, users may wish to use the GPU accelerated version of ESFS to perform ES correlation metric calculations.

By default the GPU version of ESFS will be loaded when running ``` import esfs ``` if a compatible version of CUDA is installed on the machine, and a message will print saying that the GPU version is in use.

If users wish to force ESFS to run using CPUs, they may do so by running ``` esfs.configure(gpu=False) ``` after running ``` import esfs ```.
