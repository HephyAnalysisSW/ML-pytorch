import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import os


import numpy as np
import awkward as ak

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import uproot
from tools.WeightInfo    import WeightInfo




# specify which observables to plot
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
file_names = '/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0.root:Events'
selection = '(genJet_pt>500) & (genJet_SDmass>0) & (abs(dR_genJet_maxQ1Q2b)<0.6) & (genJet_SDsubjet1_mass>=0)'
def load_data(file_names=file_names, selection=selection, coefficients=coefficients):
	# get branch names and fromat file names
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


# https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
   

