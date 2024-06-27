import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y, sample_weights, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train, sample_weight=weights_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
    return clf, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier")
    parser.add_argument("--input", type=str, help="Path to input data file")
    parser.add_argument("--output", type=str, help="Path to output pickle file")
    parser.add_argument("--shuffle", action='store_true', help="Whether to shuffle weights")
    args = parser.parse_args()
    
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    X, wX, wY = data['X'], data['wX'], data['wY']
    n = len(wX)
    labels = np.array([0] * n + [1] * n)
    weights_combined = np.concatenate([wX, wY])
    
    if args.shuffle:
        perm_indices = np.random.permutation(len(weights_combined))
        weights_combined = weights_combined[perm_indices]
    
    X_combined = np.vstack([X, X])
    clf, accuracy = train_and_evaluate(X_combined, labels, weights_combined)
    
    with open(args.output, 'wb') as f:
        pickle.dump({'accuracy': accuracy, 'model': clf}, f)

if __name__ == "__main__":
    main()

