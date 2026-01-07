# ESFS

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection and marker gene identification in single cell RNA sequencing datasets.

Go to the Example_Workflows folder to see some example workflows that you will hopefully be able to easily adapt for your own data.

### Installation
Either install this repository directly via:

```
pip install git+ssh://git@github.com/aradley/ESFS.git
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

This repository is currently in development and likely to change, but the underlying cESFW theory should be locked in at this point.

## GPU acceleration

Add description here