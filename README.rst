
.. index-inclusion-marker

collab-environment
==================

`collab-environment` is an integration package across projects for representing, modeling, and simulating behavior within 3D environments.

------------

Setup
-----
* Using pip / uv:

.. code:: sh
   pip install -e ".[dev]"


* Using conda

   We also provide a conda `env.yml` file that can be used to create a conda environment with the necessary dependencies. Run the following command to create the environment:

   .. code:: sh

      conda env create -n collab-environment -f env.yml
      conda activate collab-environment

* linux/arm64 is not supported by nvidia/label/cuda-11.8.0??

* Docker setup

   .. code:: sh

      eval $(ssh-agent)
      ssh-add ~/.ssh/id_rsa

      docker build --platform=linux/arm64  --progress=plain -t tommybotch/collab-environment .
      docker push tommybotch/collab-environment:latest

* How to painfully install nerfstudio 

   .. code:: sh

      conda env create -n nerfstudio -f env.yml
      conda activate nerfstudio

   * For some reason, this needs to be installed via pip for tiny-cuda-nn to install

      .. code:: sh
   
         pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

   * Install CUDA toolkit

      .. code:: sh

         conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

   * Need to downgrade setuptools 

      .. code:: sh

         pip install setuptools==69.5.1

   * Install tiny-cuda-nn

      .. code:: sh
   
         pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch



