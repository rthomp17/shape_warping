.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/shape_warping.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/shape_warping
    .. image:: https://readthedocs.org/projects/shape_warping/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://shape_warping.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/shape_warping/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/shape_warping
    .. image:: https://img.shields.io/pypi/v/shape_warping.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/shape_warping/
    .. image:: https://img.shields.io/conda/vn/conda-forge/shape_warping.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/shape_warping
    .. image:: https://pepy.tech/badge/shape_warping/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/shape_warping
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/shape_warping

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=============
shape_warping
=============

Code by Ondrej Biza and Skye Thompson
Tested in python 3.9

To install:

``` 
     git clone
     cd shape_warping
     pip install -e . 
```

Examples include:
 - train_warp_model.py: Trains a shape warping model from a set of pose-aligned meshes representing objects from a category 
 - train_warp_model_parts.py: Same as training a warp model, but in a loop over a set of presegmented parts 
 - shape_reconstruction_example.py: Given a shape warping model of a category, reconstruct the mesh and point cloud of an object from that category in an arbitrary pose from a partial point cloud.
 - shape_reconstruction_by_parts.py: Given a shape warping model of a set of part categories, reconstruct the mesh and point cloud of an object from that category in an arbitrary pose from presegmented partial point clouds of each part.
 - shape_generation_visualizer.py: Given one or multiple shape models, enables interactively exploring the latent space by varying individual PCA components. launched via Dash at localhost:8050.


TODOs: 
     - Expand this documentation with figs
     - Better documentation for the slider app
     - Add in size and shape regularization
     - Add function for calculating and using relational descriptors
     - Add example script for doing skill transfer
     - Incorporate code for non-PCA warping model
     - Add warp/cost comparison example

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
