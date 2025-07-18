# COSIE

<p align="left">
  <img src=./image/logo.png width="200"/>
</p>

[![python >3.9.19](https://img.shields.io/badge/python-3.9.19-blue)](https://www.python.org/) 


COSIE enables within- and **C**r**O**ss-subject **S**patial multimodal **I**ntegration, imputation, and **E**nhancement. 




<p align="center">
  <img width="95%" src=./image/framework.png>
</p>



## Usage
For detailed usage instructions and various applications of COSIE, please refer to the [COSIE Documentation](#).

Example notebooks with data are provided in the [Examples](./Examples/).



## Installation

For convenience, we recommend creating and activating a dedicated conda environment before installing COSIE.
If you haven't installed conda yet, we suggest using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), a lightweight distribution of conda.



```bash
conda create -n cosie_env python=3.9.19
conda activate cosie_env
```      

The COSIE package can be downloaded by:
```bash
git clone https://github.com/weilicode/cosie.git
cd cosie
```


The cosie_env environment can be used in jupyter notebook by:

```bash
pip install ipykernel
python -m ipykernel install --user --name=cosie_env
```


## Dependencies

COSIE is a graph deep learning model built upon [![torch-2.4.0](https://img.shields.io/badge/torch-2.4.0-orange)](https://pytorch.org/) and [![torch__geometric-2.5.3](https://img.shields.io/badge/torch__geometric-2.5.3-blueviolet)](https://pytorch-geometric.readthedocs.io/en/latest/).

Using GPU acceleration can significantly speed up the training process. If you plan to use a GPU, please make sure that PyTorch and PyTorch Geometric are installed with versions that are compatible with your local CUDA version.

All other required packages are listed in [requirements.txt](requirements.txt). You can install them by running:

```bash
pip install -r requirements.txt
```



## Questions
If you have any questions about COSIE, feel free to open an [issue](https://github.com/weilicode/cosie/issues) or contact us via email(Wei.Li@PennMedicine.upenn.edu).

