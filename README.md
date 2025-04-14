# Implementation of WGCNA for python

Comes with a number of QoL updates compared to normal WGCNA, including:

- Use of Ledoit-Wolf Shrinkage estimation.
- Demonstration of the removal of spurious
- Spectral Clustering as opposed to heirarchical clustering, with CH scoring for optimal clustering.

The tutorial notebook shown in the examples/ folder goes through how to use this notebook. Currently the implementation is designed for using the negative binomial parameters directly from several samples. However, this module can directly evaluate single cell data in the same way as the WGCNA module. 

Some possible additions in the future might be:

- Additional activation functions:
  - based on regression p-values (-log(p)) as described in the original WGCNA paper'
  - based on mutual information measures (one could try to use `sklearn.feature_selection.mutual_info_regression`)
- Better gprofiler integration (basic example is shown in the notebook given).
- More options for thresholding of spurious correlations. 
