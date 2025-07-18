Installation Guide
==================

This guide will help you install and set up the COSIE package.

We recommend creating and activating a dedicated conda environment before installation. If you haven't installed conda yet, we suggest using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_.

Create and activate the environment
------------------------------------

.. code-block:: bash

   conda create -n cosie_env python=3.9.19
   conda activate cosie_env

Clone the COSIE repository
---------------------------

.. code-block:: bash

   git clone https://github.com/weilicode/cosie.git
   cd cosie

Enable Jupyter Notebook support
--------------------------------

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=cosie_env

Dependencies
------------

COSIE is a graph deep learning model built upon:

- `torch==2.4.0 <https://pytorch.org/>`_
- `torch_geometric==2.5.3 <https://pytorch-geometric.readthedocs.io/en/latest/>`_

If you plan to use a GPU, ensure that your PyTorch and PyTorch Geometric versions match your local CUDA setup.

All additional dependencies are listed in ``requirements.txt`` and can be installed with:

.. code-block:: bash

   pip install -r requirements.txt
