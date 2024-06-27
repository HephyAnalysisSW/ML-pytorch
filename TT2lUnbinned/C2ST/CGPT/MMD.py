import numpy as np

def rff_transform(X, D, sigma=1.0):
    """Transforms data X using Random Fourier Features."""
    N, d = X.shape
    W = np.random.normal(0, 1/sigma, (d, D))
    b = np.random.uniform(0, 2*np.pi, D)
    Z = np.sqrt(2.0 / D) * np.cos(np.dot(X, W) + b)
    return Z

def compute_mmd_rff(X, Y, D, sigma=1.0):
    """Computes the MMD using Random Fourier Features."""
    Z_X = rff_transform(X, D, sigma)
    Z_Y = rff_transform(Y, D, sigma)
    
    mmd2 = np.mean(np.dot(Z_X, Z_X.T)) + np.mean(np.dot(Z_Y, Z_Y.T)) - 2 * np.mean(np.dot(Z_X, Z_Y.T))
    return mmd2

def mmd_permutation_test(X, Y, D=100, sigma=1.0, num_permutations=1000):
    """Performs the MMD permutation test."""
    observed_mmd = compute_mmd_rff(X, Y, D, sigma)
    
    combined = np.vstack([X, Y])
    n = len(X)
    permuted_mmds = []
    
    for _ in range(num_permutations):
        perm_indices = np.random.permutation(len(combined))
        perm_X = combined[perm_indices[:n]]
        perm_Y = combined[perm_indices[n:]]
        permuted_mmd = compute_mmd_rff(perm_X, perm_Y, D, sigma)
        permuted_mmds.append(permuted_mmd)
    
    permuted_mmds = np.array(permuted_mmds)
    p_value = np.mean(permuted_mmds >= observed_mmd)
    
    return observed_mmd, p_value

# Example usage with high-dimensional data
np.random.seed(42)
sample1 = np.random.randn(1000, 100)  # High-dimensional sample 1 (e.g., 100 dimensions)
sample2 = np.random.randn(1000, 100)  # High-dimensional sample 2 (e.g., 100 dimensions)

observed_mmd, p_value = mmd_permutation_test(sample1, sample2, D=200, sigma=1.0)
print(f"Observed MMD: {observed_mmd}, p-value: {p_value}")

