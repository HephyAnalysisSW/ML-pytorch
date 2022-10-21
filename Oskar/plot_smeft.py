import sys
sys.path.insert(0, '..')

import os
plot_directory = '/groups/hephy/cms/oskar.rothbacher/www/pytorch/genTops/genTops'

import numpy as np
import awkward as ak

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import uproot
from tools.WeightInfo    import WeightInfo

import matplotlib.pyplot as plt



# specify whichobservables to plot
def get_branch_names():
	from branches import branches
	with uproot.open('/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0.root:Events') as f:
		scalar_branches=f.keys(branches, filter_typename=['float', 'int32_t','uint64_t'])
		vector_branches=f.keys(branches, filter_typename='*[]')
	return scalar_branches, vector_branches


## return the subset of combinations that include at most the elements of coefficients
def make_combinations(coefficients):
    combinations = []
    for comb in w.combinations:
        good = True
        for k in comb:
            if k not in coefficients:
                good = False
                break
        if good:
            combinations.append(comb)
    return combinations

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/t-sch-RefPoint-noWidthRW_reweight_card.pkl'
w = WeightInfo(reweight_pkl)
w.set_order(2)

# read from root files and filter, specify eft coefficients
coefficients = ['ctWRe',]
file_names = '/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0.root'
selection = '(genJet_pt>500) & (genJet_SDmass>0) & (abs(dR_genJet_maxQ1Q2b)<0.6) & (genJet_SDsubjet1_mass>=0)'
def load_data(file_names=file_names, selection=selection, coefficients=coefficients):
	# get branch names and fromat file names
	file_names = file_names+':Events'
	scalar_branches, vector_branches = get_branch_names()
	# load scalar and vector branches as awkward arrays
	scalar_events = uproot.concatenate(file_names, cut=selection, branches=scalar_branches)							
	vector_events = uproot.concatenate(file_names, cut=selection, branches=vector_branches)
	# load the weights 
	p_C = uproot.concatenate(file_names, cut=selection, branches='p_C')
	p_C = ak.to_numpy(p_C.p_C)
	combinations = make_combinations(coefficients)
	weights = np.array([p_C[:,w.combinations.index(comb)] for comb in combinations]).transpose()
	#convert to numpy
	scalar_events = np.array([scalar_events[sb] for sb in scalar_branches]).transpose().astype('float32')
	for vb in vector_branches:
	# print(vb)
		max = 1
		for ve in vector_events[vb]:
			if len(ve)>max: max=len(ve)
		vector_events[vb] = ak.fill_none(ak.pad_none(vector_events[vb], target=max, clip=True), value=0)
		np.stack

	return scalar_events, vector_events, weights


# get the weights
def weight_theta(weights, theta):
    weight_theta =  weights[:,0] * (1
                                    + theta * weights[:,1] / weights[:,0] 
                                    #+ 0.5 * theta**2 * weights[('ctWRe','ctWRe')] / weights[()]
                                    )
    return weight_theta

# join the data and the labels in a jointdataset
class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], *tuple(y[idx] for y in self.y)

# class JointDataset(Dataset):
#     def __init__(self, x1, x2, y):
#         self.x1 = x1
#         self.x2 = x2
#         self.y = y

#     def __len__(self):
#         return len(self.x1)

#     def __getitem__(self, idx):
#         return self.x1[idx], self.x2[idx], self.y[idx]


# plot function
def plot_eft_hists(data, branch_list, weights, bins=50, theta=(-1.0,1.0)):
	n_plots = int(np.ceil(np.sqrt(len(branch_list))))
	plt.subplots(figsize=[10*n_plots,10*n_plots])
	for n, branch in enumerate(branch_list):
		plt.subplot(n_plots,n_plots,n+1)
		plt.hist(data[:,n], bins=bins, weights=weights[:,0],
					histtype='step', color='black', label='ctW=0', density=True)
		for th in theta:
			plt.hist(data[:,n], bins=bins, weights=weight_theta(weights, th),
						histtype='step', label=f'ctW={th}', density=True)
		plt.title(branch)
		plt.legend()
	plt.savefig(os.path.join(plot_directory,'hists.png'))
	print(f'saved file to {os.path.join(plot_directory,"hists.png")}')










# # get the features from the branches of interest
# # features = np.stack([events[branch].to_numpy() for branch in branches], axis = 1)


