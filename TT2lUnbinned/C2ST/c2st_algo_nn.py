import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

def BCEWeightedLoss( outputs, wX, wY ):
    o = outputs[:,0]
    mask = (o>0).bool() & (o<1).bool()
    return -wY[mask]*torch.log(o[mask]) - wX[mask]*torch.log(1.-o[mask]) 

def c2st_test(X, wX, wY, test_size=0.3, random_state=42, num_epochs=100, batch_size=512):
    """Performs the C2ST for weighted samples."""
    # Create labels for the samples
    n = len(wX)
    
    # Split into train and test sets
    X_train, X_test, wX_train, wX_test, wY_train, wY_test = train_test_split(
        X, wX, wY, test_size=test_size, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train  = torch.tensor(X_train, dtype=torch.float32)
    wX_train = torch.tensor(wX_train, dtype=torch.float32)
    wY_train = torch.tensor(wY_train, dtype=torch.float32)
    X_test   = torch.tensor(X_test, dtype=torch.float32)
    wX_test  = torch.tensor(wX_test, dtype=torch.float32)
    wY_test  = torch.tensor(wY_test, dtype=torch.float32)

    # Create the model
    model = SimpleNN(input_dim=X_train.shape[1])
    #criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Training loop with progress bar
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        #permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            #indices = permutation[i:i + batch_size]
            batch_X, batch_wX, batch_wY = X_train[i:i + batch_size], wX_train[i:i + batch_size], wY_train[i:i + batch_size]

            # Forward pass
            outputs = model(batch_X)
            #loss = criterion(outputs, batch_y)
            #loss = (loss * batch_weights).mean()

            #print ("batch_X",batch_X)    
            #print ("batch_wX",batch_wX)    
            #print ("batch_wY",batch_wY)    
            #print ("outputs",outputs)    

            loss = BCEWeightedLoss( outputs, batch_wX, batch_wY ).sum()

            #print ("loss", loss)
            #assert False, ""
     
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_train)
            predicted = (outputs[:,0] >= 0.5).bool()
            accuracy = (wX_train[~predicted].sum() + wY_train[predicted].sum())/( wX_train.sum() + wY_train.sum())
            #accuracy_score(y_train.numpy(), predicted.numpy(), sample_weight=weights_train.numpy())
            loss = BCEWeightedLoss( outputs, wX_train, wY_train ).sum()
            print ("train accuracy", accuracy, "loss", loss)
            #print (">1", torch.count_nonzero(outputs[:,0]>1), "<0", torch.count_nonzero(outputs[:,0]<0))
            outputs = model(X_test)
            predicted = (outputs[:,0] >= 0.5).bool()
            accuracy = (wX_test[~predicted].sum() + wY_test[predicted].sum())/( wX_test.sum() + wY_test.sum())
            print ("test accuracy", accuracy)

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs[:,0] >= 0.5).bool()
        accuracy = (wX_test[~predicted].sum() + wY_test[predicted].sum())/( wX_test.sum() + wY_test.sum())
        print ("accuracy", accuracy)

    return model, accuracy

    ## Predict on the test set
    #y_pred = clf.predict(X_test)
    #
    ## Compute the accuracy
    #accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
    #
    ## Null hypothesis: samples are from the same distribution, expect accuracy around 0.5
    ## Alternative hypothesis: samples are from different distributions, expect accuracy > 0.5
    #return clf, accuracy

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

