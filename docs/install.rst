Installation Guide
==================

This guide will help you install and set up the COSIE package.

Installation
~~~~~~~~~~~~~~~~

For convenience, we recommend creating and activating a dedicated conda environment before installing COSIE.
If you haven't installed conda yet, we suggest using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_, a lightweight distribution of conda.

.. code-block:: bash

   conda create -n cosie_env python=3.9.19
   conda activate cosie_env

The COSIE package can be downloaded by:

.. code-block:: bash

   git clone https://github.com/weilicode/cosie.git
   cd cosie

The ``cosie_env`` environment can be used in Jupyter Notebook by:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=cosie_env


Dependencies
~~~~~~~~~~~~~~~~

COSIE is a graph deep learning model built upon `torch-2.4.0 <https://pytorch.org/>`_ and 
`torch_geometric-2.5.3 <https://pytorch-geometric.readthedocs.io/en/latest/>`_.

Using GPU acceleration can significantly speed up the training process. If you plan to use a GPU, please make sure that PyTorch and PyTorch Geometric are installed with versions that are compatible with your local CUDA version. For example, if you are using CUDA 12.1, you can install the required packages as follows:

.. code-block:: bash

   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install torch_geometric==2.5.3

All other required packages are listed in requirements.txt. You can install them by running:

.. code-block:: bash

   pip install -r requirements.txt
