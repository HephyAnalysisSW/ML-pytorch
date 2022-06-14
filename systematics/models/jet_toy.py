alpha = 50.
jec   = 0.05
import numpy as np
max_NJets = 1

#thresholds = [20,40,10**6]
#thresholds = [20,10**6]

def getEvents( NEvents = 100000, nBins=None):

    # array of exp numbers, reshape to two more
    #jets = np.random.exponential( scale=alpha, size=(3+max_NJets)*NEvents).reshape(NEvents,-1)
    jets = np.random.exponential( scale=alpha, size=(max_NJets)*NEvents).reshape(NEvents,-1)
    # sort axis=1, select max_NJets columns 
    np.matrix.sort(jets)
    jets=np.flip(jets)[:, :max_NJets]

    jets = {0:jets, -1:(1-jec)*jets, +1:(1+jec)*jets}
    # remove low pt jets
    for k, j in jets.items():
        j[j<20]=-1
    # remove those events where we have too few jets
        jets[k]  = j[(j>0).all(axis=1)]
        #jets[k]  = np.ones_like(jets[k])
        if nBins is not None:
            thresholds = np.exp(np.linspace(np.log(20),np.log(alpha*nBins),nBins+1))
            thresholds[-1]=float('inf')
            jets[k]  = np.digitize(jets[k], thresholds)-1
    return jets

features = ["jet_pt_%i"%i for i in range(max_NJets)]
#plot_options = {"jet_pt_%i"%i:{'tex':'p_{T}(j_{%i})'%i, 'binning':[50,20,520]} for i in range(max_NJets)}
plot_options = {"jet_pt_%i"%i:{'tex':'p_{T}(j_{%i})'%i, 'binning':[3,0,3]} for i in range(max_NJets)}
