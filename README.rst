
.. index-inclusion-marker

splitspat
========

SplitSplat is a wrapper for `nerfstudio <https://github.com/nerfstudio-project/nerfstudio/>`_ that allows generation of 3D scenes from images/videos using different NeRF and Gaussian Splatting methods.

Installation
------------

We provide a conda env.yml file that can be used to create a conda environment with the necessary dependencies. Run the following command to create the environment:

.. code:: sh

   conda env create -n splat -f env.yml
   conda activate splat

docker build --platform=linux/amd64,linux/arm64 -t tommybotch/collab-environment .