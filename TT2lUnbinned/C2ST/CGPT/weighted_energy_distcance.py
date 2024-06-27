import numpy as np
from scipy.spatial.distance import cdist

def chunked_weighted_sum(X, Y, wX, wY, chunk_size=100):
    n, m = len(X), len(Y)
    T1 = 0.0
    T2 = 0.0
    T3 = 0.0
    
    sum_wX = np.sum(wX)
    sum_wY = np.sum(wY)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, m, chunk_size):
            end_j = min(j + chunk_size, m)
            dXY_chunk = cdist(X[i:end_i], Y[j:end_j], 'euclidean')
            T1 += np.sum(dXY_chunk * np.outer(wX[i:end_i], wY[j:end_j]))
            
    T1 *= 2 / (sum_wX * sum_wY)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        dXX_chunk = cdist(X[i:end_i], X, 'euclidean')
        T2 += np.sum(dXX_chunk * np.outer(wX[i:end_i], wX))
    
    T2 /= sum_wX**2
    
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        dYY_chunk = cdist(Y[i:end_i], Y, 'euclidean')
        T3 += np.sum(dYY_chunk * np.outer(wY[i:end_i], wY))
    
    T3 /= sum_wY**2
    
    return T1 - T2 - T3

def weighted_energy_distance(X, Y, wX, wY, chunk_size=100):
    return chunked_weighted_sum(X, Y, wX, wY, chunk_size)

def permutation_test(X, Y, wX, wY, num_permutations=1000, chunk_size=100):
    observed_stat = weighted_energy_distance(X, Y, wX, wY, chunk_size)
    combined = np.vstack([X, Y])
    combined_weights = np.hstack([wX, wY])
    
    n = len(X)
    permuted_stats = []
    for i_perm in range(num_permutations):
        if i_perm%100==0: 
            print ("permutation", i_perm)
        perm_indices = np.random.permutation(len(combined))
        perm_X = combined[perm_indices[:n]]
        perm_Y = combined[perm_indices[n:]]
        perm_wX = combined_weights[perm_indices[:n]]
        perm_wY = combined_weights[perm_indices[n:]]
        permuted_stat = weighted_energy_distance(perm_X, perm_Y, perm_wX, perm_wY, chunk_size)
        permuted_stats.append(permuted_stat)
    
    permuted_stats = np.array(permuted_stats)
    p_value = np.mean(permuted_stats >= observed_stat)
    
    return observed_stat, p_value
if __name__=="__main__":
    # Example usage:
    np.random.seed(44)
    Nevent  = 100
    sample1 = np.random.randn(Nevent, 3)  # Example high-dimensional sample 1
    sample2 = sample1 #np.random.randn(Nevent, 3)  # Example high-dimensional sample 2
    weights1 = np.random.rand(Nevent)      # Example weights for sample 1
    weights2 = np.random.rand(Nevent)      # Example weights for sample 2

    weights2*=np.exp(0.5*sample2[:,0])

    observed_stat, p_value = permutation_test(sample1, sample2, weights1, weights2)
    print(f"Observed Weighted Energy Distance: {observed_stat}, p-value: {p_value}")

