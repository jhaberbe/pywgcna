import numpy as np

def find_scale_free_power(x: np.array) -> float:
    """Compute Scale Free Power using heuristic log(connectivity) vs log(p(k))

    Args:
        corr_coef (np.array): Pairwise Pearson's correlation

    Returns:
        int: smallest power that induces a scale free topology (R^2 of 0.8).
    """
    # Create a copy of the original correlation coefficient matrix
    corr_coef_copy = x.copy()

    # Initialize variables
    r2_max = 0
    power_max = 0

    # Iterate until r2 reaches 0.8 or power exceeds 12
    for power in range(1, 21):

        # Compute the histogram of the logarithm of k
        n, k = np.histogram(np.nansum(corr_coef_copy, axis=0), bins=10)
        n += 1
        k += 1

        logk = np.log10(k)

        # Normalize the histogram and calculate the squared correlation coefficient (r2)
        logp = np.log10(n / n.sum())
        r2 = np.corrcoef(logk[1:], logp)[0, 1] ** 2

        # Update r2_max and power_max if r2 is the highest observed so far
        if r2 > r2_max:
            r2_max = r2
            power_max = power
            print(f"Power: {power}, R^2: {r2}")

        # If r2 has reached the threshold, break early
        if r2 >= 0.8:
            return power_max

        # Update the correlation coefficient matrix for the next iteration
        corr_coef_copy *= x
    
    return power_max

def topological_overlap(A: np.array) -> np.array:
    """
    Topological overlap function.

    Parameters
    ----------
    x : np.array
        Input array.

    Returns
    -------
    np.array
        Topological overlap value of the input array.
    """
    # Number of nodes
    n = A.shape[0]
    
    # Compute the degree of each node
    k = np.sum(A, axis=1)
    
    # Compute the l_ij (pairwise connectivity) matrix using matrix multiplication
    L = A @ A
    
    # Add the diagonal of A to L to get l_ij + A_ij
    L_plus_A = L + A
    
    # Compute min(k_i, k_j) matrix
    min_k = np.minimum.outer(k, k)
    
    # Compute TOM matrix
    TOM = L_plus_A / (min_k + 1 - A)
    
    return TOM
