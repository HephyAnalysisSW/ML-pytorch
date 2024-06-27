import numpy as np

def rff_transform(X, D, sigma=1.0):
    """Transforms data X using Random Fourier Features."""
    N, d = X.shape
    W = np.random.normal(0, 1/sigma, (d, D))
    b = np.random.uniform(0, 2*np.pi, D)
    Z = np.sqrt(2.0 / D) * np.cos(np.dot(X, W) + b)
    return Z

def weighted_mmd_rff(X, wX, wY, D, sigma=1.0):
    """Computes the weighted MMD using Random Fourier Features."""
    Z = rff_transform(X, D, sigma)
    
    weighted_mean_X = np.sum(Z.T * wX, axis=1) / np.sum(wX)
    weighted_mean_Y = np.sum(Z.T * wY, axis=1) / np.sum(wY)
    
    mmd2 = np.sum((weighted_mean_X - weighted_mean_Y)**2)
    return mmd2

def weighted_mmd_permutation_test(X, wX, wY, D=100, sigma=1.0, num_permutations=1000):
    """Performs the weighted MMD permutation test."""
    observed_mmd = weighted_mmd_rff(X, wX, wY, D, sigma)
    
    permuted_mmds = []
    for _ in range(num_permutations):
        perm_indices = np.random.permutation(len(wX))
        perm_wX = wX[perm_indices]
        perm_wY = wY[perm_indices]
        permuted_mmd = weighted_mmd_rff(X, perm_wX, perm_wY, D, sigma)
        permuted_mmds.append(permuted_mmd)
    
    permuted_mmds = np.array(permuted_mmds)
    p_value = np.mean(permuted_mmds >= observed_mmd)
    
    return observed_mmd, p_value

# Example usage with weighted samples
np.random.seed(42)

Nevents = 10**5

sample = np.random.randn(Nevents, 100)  # High-dimensional sample (e.g., 100 dimensions)
weights1 = np.random.rand(Nevents)      # Weights for the first sample
weights2 = np.random.rand(Nevents)      # Weights for the second sample

weights2*=np.exp(0.2*sample[:,0])

observed_mmd, p_value = weighted_mmd_permutation_test(sample, weights1, weights2, D=200, sigma=1.0)
print(f"Observed Weighted MMD: {observed_mmd}, p-value: {p_value}")

