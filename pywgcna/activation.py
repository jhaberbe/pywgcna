import numpy as np

def absolute_activation(x: np.array) -> np.array:
    """
    Absolute activation function.

    Parameters
    ----------
    x : np.array
        Input array.

    Returns
    -------
    np.array
        Absolute value of the input array.
    """
    return np.abs(x)

def hybrid_activation(x: np.array) -> np.array:
    """
    Hybrid activation function.

    Parameters
    ----------
    x : np.array
        Input array.

    Returns
    -------
    np.array
        Hybrid activation value of the input array.
    """
    return (((1+x) - np.eye(x.shape[0])) / 2)

def strictly_positive_activation(x: np.array) -> np.array:
    """
    Strict activation function.

    Parameters
    ----------
    x : np.array
        Input array.

    Returns
    -------
    np.array
        Strictly positive activation value of the input array.
    """
    x = (x - np.eye(x.shape[0]))
    return np.where(x > 0, x, 0)