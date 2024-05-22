import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

def c2st_test(X, wX, wY, test_size=0.3, random_state=42):
    """Performs the C2ST for weighted samples."""
    # Create labels for the samples
    n = len(wX)
    labels = np.array([0] * n + [1] * n)
    
    # Combine the samples
    X_combined = np.vstack([X, X])
    weights_combined = np.concatenate([wX, wY])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X_combined, labels, weights_combined, test_size=test_size, random_state=random_state
    )
    
    # Train the classifier with sample weights
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train, sample_weight=weights_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Compute the accuracy
    accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
    
    # Null hypothesis: samples are from the same distribution, expect accuracy around 0.5
    # Alternative hypothesis: samples are from different distributions, expect accuracy > 0.5
    return clf, accuracy

if __name__=="__main__":

    Nevents = 1000000

    # Example usage with weighted samples
    np.random.seed(42)
    sample = np.random.randn(Nevents, 100)  # High-dimensional sample (e.g., 100 dimensions)
    weights1 = np.random.rand(Nevents)      # Weights for the first sample
    weights2 = np.random.rand(Nevents)      # Weights for the second sample
        
    weights2*=np.exp(0.02*sample[:,0])

    accuracy = c2st_test(sample, weights1, weights2, test_size=0.3, random_state=42)
    print(f"Classifier Accuracy: {accuracy}")

    # A significance test can be performed to check if the accuracy is significantly higher than 0.5
    from scipy.stats import binom_test
    n_test_samples = int(0.3 * len(sample) * 2)  # Number of test samples
    p_value = binom_test(int(accuracy * n_test_samples), n_test_samples, p=0.5, alternative='greater')
    print(f"p-value: {p_value}")

