import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def compute_covariance(x: np.array) -> np.array:
    """
    Compute the correlation matrix of the input array.

    Parameters
    ----------
    x : np.array
        Input array.

    Returns
    -------
    np.array
        Correlation matrix of the input array.
    """
    return np.corrcoef(x) 

def compute_shrunk_correlation(x: np.array) -> np.array:
    """
    Compute the shrunk covariance matrix of the input array using Ledoit Wolf Estimator.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Shrunk covariance matrix of the input array.
    """
    # Compute the covariance matrix
    lw = LedoitWolf()
    lw.fit(x)

    # Compute the shrunk covariance matrix
    shrunk_cov = lw.covariance_

    # Compute the correlation matrix R_ij = ( sigma_ij / sqrt(sigma_ii * sigma_jj) )
    d = np.sqrt(np.diag(shrunk_cov))
    shrunk_corr = shrunk_cov / np.outer(d, d)
    shrunk_corr[np.isnan(shrunk_corr)] = 0
    np.fill_diagonal(shrunk_corr, 0)
    return shrunk_corr
