# Overview
This is the documentation repository for the clustering algorithm of the paper "Interpretable Clustering: An Optimization Approach" by Dimitris Bertsimas, Agni Orfanoudaki, and Holly Wiberg. The purpose of this method, ICOT, is to generate interpretable tree-based clustering models. 

# Academic License and Installation

This code runs in Julia version 1.1.0, which can be downloaded at the following links:
* Linux: https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz
* Mac: https://julialang-s3.julialang.org/bin/mac/x64/1.1/julia-1.1.0-mac64.dmg

*Note: version 1.1.0 is required for compatibility with the package.*

The ICOT software package uses tools from the [Interpretable AI](https://www.interpretable.ai/) suite and thus it requires an academic license. 

You can download the system image the following links:
* [Linux](https://iai-system-images.s3.amazonaws.com/icot/linux/julia1.1.0/v1.0/sys-linux-julia1.1.0-iai0.1.0-878.zip) 
* [Mac](https://iai-system-images.s3.amazonaws.com/icot/macos/julia1.1.0/v1.0/sys-macos-julia1.1.0-iai0.1.0-878.zip)

You can find detailed installation guidelines for the system image [here](https://docs.interpretable.ai/stable/installation/).

Once you have completed the installation you will be presented with a machine ID. You can request an academic license by emailing <info@interpretable.ai> with your academic institution address and the subject line "Request for ICOT License". Please include the machine ID in the email, and Interpretable AI will generate a license file for your machine.

# Algorithm Guidelines

The main command to run the algorithm on a dataset `X` is `ICOT.fit!(learner, X, y);` where the `y` can refer to some data partition that is associated with the dataset. The `learner` is defined as an `ICOT.InterpretableCluster()` object with the following parameters:
* `criterion`: defines the internal validation criterion used to train the ICOT algorithm. The algorithm accepts to options `:dunnindex` ([Dunn 1974](https://www.tandfonline.com/doi/abs/10.1080/01969727408546059)) and `:silhouette` ([Rousseeuw 1987](https://www.sciencedirect.com/science/article/pii/0377042787901257)).
* `ls_warmstart_criterion`: defines the internal validation criterion used to create the initial solution of the warmstart. The same options are offered with the `criterion` parameter.
* `kmeans_warmstart`: provides a warmstart solution to initialize the algorithm. Details are provided in Section 3.3.2 of the [paper](https://link.springer.com/article/10.1007/s10994-020-05896-2). It can take as input `:none`, `:greedy`, and `:oct`. The OCT option uses user-selected labels (i.e. from K-means) to fit an Optimal Classification Tree as a supervised learning problem to provide a warm-start to the algorithm. The greedy option fits a CART tree to these labels.
* `geom_search`: is a boolean parameter that controls where the algorithm will enable the geometric component of the feature space search. See details in Section 3.3.1 of the [paper](https://link.springer.com/article/10.1007/s10994-020-05896-2).
* `geom_threshold`: refers to the percentile of gaps that will be considered by the geometric search for each feature. For example: 0.99. 
* `minbucket`: controls the minimum number of points that must be present in every leaf node of the fitted tree. 
* `max_depth`: accepts a non-negative Integer to control the maximum depth of the fitted tree. This parameter must always be explicitly set or tuned. We recommend tuning this parameter using the grid search process described in the guide to parameter tuning.
* `ls_random_seed`: is an integer controlling the randomized state of the algorithm. We recommend to set the seed to ensure reproducability of results.
* `ls_num_tree_restarts`: is an integer specifying the number of random restarts to use in the local search algorithm. Must be positive and defaults to 100. The performance of the tree typically increases as this value is increased, but with quickly diminishing returns. The computational cost of training increases linearly with this value. 
* `cp`:  the complexity parameter that determines the tradeoff between the accuracy and complexity of the tree to control overfitting, as commonly seen in supervised learning problems. The internal validation criteria used for this unsupervised algorithm naturally limit the tree complexity, we recommend to set the value to 0.0.

You can visualize your model on a browser using the `ICOT.showinbrowser()` command.

You can evaluate the score on a trained ICOT model using the `score_al_ws_oct = ICOT.score(learner, X, y, criterion);` command.

We have added an example for the ruspini dataset in the `src` folder called `runningICOT_example.jl`.

# Citing ICOT
If you use ICOT in your research, we kindly ask that you reference the original [paper](https://link.springer.com/article/10.1007/s10994-020-05896-2) that first introduced the algorithm:

```
@article{bertsimas2020interpretable,
  title={Interpretable clustering: an optimization approach},
  author={Bertsimas, Dimitris and Orfanoudaki, Agni and Wiberg, Holly},
  journal={Machine Learning},
  pages={1--50},
  year={2020},
  publisher={Springer}
}
```

