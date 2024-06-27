import os
import subprocess
import pickle
import numpy as np

def main():
    np.random.seed(42)
    sample = np.random.randn(1000, 100)  # High-dimensional sample (e.g., 100 dimensions)
    weights1 = np.random.rand(1000)      # Weights for the first sample
    weights2 = np.random.rand(1000)      # Weights for the second sample
    
    data = {'X': sample, 'wX': weights1, 'wY': weights2}
    
    input_file = 'input_data.pkl'
    with open(input_file, 'wb') as f:
        pickle.dump(data, f)
    
    num_permutations = 10
    accuracies = []
    
    # Submit the observed job
    observed_output_file = 'observed_result.pkl'
    print(['python', 'train_job.py', '--job', 'observed', '--input', input_file, '--output', observed_output_file])
    subprocess.run(['python', 'train_job.py', '--input', input_file, '--output', observed_output_file])
    
    # Submit permutation jobs
    for i in range(num_permutations):
        output_file = f'result_{i}.pkl'
        print(['python', 'train_job.py', '--job', str(i), '--input', input_file, '--output', output_file, '--shuffle'])
        subprocess.run(['python', 'train_job.py', '--input', input_file, '--output', output_file, '--shuffle'])
        accuracies.append(output_file)
    
    # Read the observed accuracy
    with open(observed_output_file, 'rb') as f:
        observed_data = pickle.load(f)
    observed_accuracy = observed_data['accuracy']
    
    # Read the permuted accuracies
    permuted_accuracies = []
    for output_file in accuracies:
        with open(output_file, 'rb') as f:
            data = pickle.load(f)
            permuted_accuracies.append(data['accuracy'])
    
    permuted_accuracies = np.array(permuted_accuracies)
    p_value = np.mean(permuted_accuracies >= observed_accuracy)
    
    print(f"Observed Accuracy: {observed_accuracy}")
    print(f"p-value: {p_value}")

if __name__ == "__main__":
    main()

