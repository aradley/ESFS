# ESFS

ESFS is an Entropy Sorting based feature selection package primarily developed for feature selection of single cell RNA sequencing datasets.

This repository is currently in development and likely to change, but the underlying cESFW theory should be locked in at this point.

Go to the Example_Workflows folder to see some example workflows that you will hopefully be able to easily adapt for your own data.

### Installation
1. Retreive the ripository with: git clone git@github.com:aradley/ESFS.git

conda create -n ESFS_Env python=3.11
conda activate ESFS_Env
conda install -c conda-forge numpy=1.26.4 scipy=1.11.4 pandas=2.2.3 matplotlib=3.7.3 scikit-learn=1.5.2 umap-learn=0.5.7 multiprocess p-tqdm anndata plotly scanpy ipywidgets notebook jupyterlab python-igraph leidenalg fastcluster

