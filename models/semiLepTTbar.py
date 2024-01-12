import pickle
import random
import ROOT
from math import pi
if __name__=="__main__":
    import sys
    sys.path.append('..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

selection = lambda ar: (ar.nrecoLep>=1) & (ar.delphesJet_dR_hadTop_maxq1q2b<0.8) & (ar.parton_hadTop_pt<1000) & (ar.parton_lepTop_b_pt<400)

feature_names = [   
            "parton_hadTop_decayAngle_theta", "parton_hadTop_decayAngle_phi", 
            "parton_hadTop_pt", "parton_hadTop_eta", "parton_hadTop_phi", "parton_hadTop_q1_pt", "parton_hadTop_q1_eta", "parton_hadTop_q2_pt", "parton_hadTop_q2_eta", "parton_hadTop_b_pt", "parton_hadTop_b_eta", "parton_hadTop_W_pt", "parton_hadTop_W_eta", 
            "parton_lepTop_pt", "parton_lepTop_eta", "parton_lepTop_phi", "parton_lep_pt", "parton_lep_eta", "parton_lep_phi", "parton_nu_pt", "parton_nu_eta", "parton_nu_phi", "parton_lepTop_b_pt", "parton_lepTop_b_eta", "parton_lepTop_b_phi", "parton_lepTop_W_pt", "parton_lepTop_W_eta", "parton_lepTop_W_phi", 
            "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", 
            "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",
    ]
observers = []

data_generator =  DataGenerator(
    input_files = ["/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/TT01jDebug_*.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection   = selection,
        branches = [
            "p_C",
            "nrecoLep", "recoLep_pt", "delphesJet_dR_hadTop_maxq1q2b", "nrecoJet", "nBTag",
            "parton_hadTop_decayAngle_theta", "parton_hadTop_decayAngle_phi", 
            "parton_hadTop_pt", "parton_hadTop_eta", "parton_hadTop_phi", "parton_hadTop_q1_pt", "parton_hadTop_q1_eta", "parton_hadTop_q2_pt", "parton_hadTop_q2_eta", "parton_hadTop_b_pt", "parton_hadTop_b_eta", "parton_hadTop_W_pt", "parton_hadTop_W_eta", 
            "parton_lepTop_pt", "parton_lepTop_eta", "parton_lepTop_phi", "parton_lep_pt", "parton_lep_eta", "parton_lep_phi", "parton_nu_pt", "parton_nu_eta", "parton_nu_phi", "parton_lepTop_b_pt", "parton_lepTop_b_eta", "parton_lepTop_b_phi", "parton_lepTop_W_pt", "parton_lepTop_W_eta", "parton_lepTop_W_phi", 
            "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", 
            "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",
        ]) 

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/TT01jDebug_reweight_card.pkl'
weightInfo = WeightInfo(reweight_pkl)
weightInfo.set_order(2)
default_eft_parameters = {p:0 for p in weightInfo.variables}
def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in weightInfo.variables+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

random_eft = make_eft(**{v:random.random() for v in weightInfo.variables} )
sm         = make_eft()

wilson_coefficients = weightInfo.variables 

def make_combinations( coefficients ):
    combinations = []
    for comb in weightInfo.combinations:
        good = True
        for k in comb:
            if k not in coefficients:
                good = False
                break
        if good:
            combinations.append(comb)
    return combinations

def getEvents( nTraining, return_observers = False):
    data_generator.load(-1, small=nTraining )
    combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch('p_C')
    if return_observers:
        return data_generator.scalar_branches( feature_names ), {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}, data_generator.scalar_branches( observers )
    else:
        return data_generator.scalar_branches( feature_names ), {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

tex = {"ctWRe":"C_{tW}^{Re}", "ctWIm":"C_{tW}^{Im}", "ctGRe":"C_{tG}^{Re}", "ctGIm":"C_{tG}^{Im}", "ctBIm":"C_{tB}^{Im}", "ctBRe":"C_{tB}^{Re}", "cHt":"C_{Ht}", 'cHtbRe':'C_{Htb}^{Re}', 'cHtbIm':'C_{Htb}^{Im}', 'cHQ3':'C_{HQ}^{(3)}'}

#['ctWRe', 'ctBRe', 'cHQ3', 'cHt', 'cHtbRe', 'ctWIm', 'ctBIm', 'cHtbIm']

plot_points = [
    {'color':ROOT.kBlack,       'point':sm, 'tex':"SM"},

    {'color':ROOT.kMagenta-4,   'point':make_eft(ctWRe=-1),  'tex':"Re(c_{tW})=-1", },
    {'color':ROOT.kMagenta+2,   'point':make_eft(ctWRe=1),   'tex':"Re(c_{tW})=1",  },
    {'color':ROOT.kGreen-4,     'point':make_eft(ctGRe=-1),  'tex':"Re(c_{tG})=-1", },
    {'color':ROOT.kGreen+2,     'point':make_eft(ctGRe=1),   'tex':"Re(c_{tG})=1",  },
    {'color':ROOT.kBlue-4,      'point':make_eft(ctGIm=-1),  'tex':"Im(c_{tG})=-1", },
    {'color':ROOT.kBlue+2,      'point':make_eft(ctGIm=1),   'tex':"Im(c_{tG})=1",  },
    ]

plot_options =  {
    "parton_hadTop_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t)'},
    "parton_hadTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t)'},
    "parton_hadTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t)'},
    "parton_lepTop_pt" :{'binning':[30,0,800], 'tex':'p_{T}(t lep)'},
    "parton_lepTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t lep)'},
    "parton_lepTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t lep)'},
    "parton_lep_pt" :{'binning':[30,0,800], 'tex':'p_{T}(l (t lep))'},
    "parton_lep_eta" :{'binning':[30,-3,3], 'tex':'#eta(l(t lep))'},
    "parton_lep_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(l(t lep))'},
    "parton_nu_pt" :{'binning':[30,0,800], 'tex':'p_{T}(#nu (t lep))'},
    "parton_nu_eta" :{'binning':[30,-3,3], 'tex':'#eta(#nu(t lep))'},
    "parton_nu_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(#nu(t lep))'},
    "parton_lepTop_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t lep))'},
    "parton_lepTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t lep))'},
    "parton_lepTop_b_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(b(t lep))'},
    "parton_lepTop_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t lep))'},
    "parton_lepTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t lep))'},
    "parton_lepTop_W_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(W(t lep))'},


    "parton_hadTop_decayAngle_theta" :{'binning':[30,0,pi], 'tex':'#theta(t had)'},
    "parton_hadTop_decayAngle_phi"   :{'binning':[30,-pi,pi], 'tex':'#phi(t had)'},

    "parton_hadTop_q1_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{1}(t had))'},
    "parton_hadTop_q1_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{1}(t had))'},
    "parton_hadTop_q2_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{2}(t had))'},
    "parton_hadTop_q2_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{2}(t had))'},
    "parton_hadTop_b_pt" :{'binning':[30,0,800], 'tex':'p_{T}(b(t had))'},
    "parton_hadTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t had))'},
    "parton_hadTop_W_pt" :{'binning':[30,0,800], 'tex':'p_{T}(W(t had))'},
    "parton_hadTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t had))'},

    "parton_cosThetaPlus_n"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{n}'},
    "parton_cosThetaMinus_n"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{n}'},
    "parton_cosThetaPlus_r"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{r}'},
    "parton_cosThetaMinus_r"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{r}'},
    "parton_cosThetaPlus_k"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{k}'},
    "parton_cosThetaMinus_k"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{k}'},
    "parton_cosThetaPlus_r_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{n}'},
    "parton_cosThetaMinus_r_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{n}'},
    "parton_cosThetaPlus_k_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{k}'},
    "parton_cosThetaMinus_k_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{k}'},
    "parton_xi_nn"              :{'binning':[30,-1,1], 'tex':'#xi_{nn}'},
    "parton_xi_rr"              :{'binning':[30,-1,1], 'tex':'#xi_{rr}'},
    "parton_xi_kk"              :{'binning':[30,-1,1], 'tex':'#xi_{kk}'},
    "parton_xi_nr_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{+}'},
    "parton_xi_nr_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{-}'},
    "parton_xi_rk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{+}'},
    "parton_xi_rk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{-}'},
    "parton_xi_nk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{+}'},
    "parton_xi_nk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{-}'},
    "parton_cos_phi"            :{'binning':[30,-1,1], 'tex':'cos(#phi)'},
    "parton_cos_phi_lab"        :{'binning':[30,-1,1], 'tex':'cos(#phi lab)'},
    "parton_abs_delta_phi_ll_lab":{'binning':[30,0,pi], 'tex':'|#Delta(#phi(l,l))|'},

}

multi_bit_cfg = {'n_trees': 300,
                 'max_depth': 4,
                 'learning_rate': 0.20,
                 'min_size': 25 }

if __name__=="__main__":
   
    # load some events and their weights 
    x, w = getEvents(1000)

    # x are a list of feature-vectors such that x[0] are the features of the first event. Their branch-names are stored in feature_names.
    # w are a dictionary with the weight-coefficients. The key tuple(), i.e., the empty n-tuple, is the constant term. The key ('ctWRe', ), i.e., the coefficient 
    # that is an tuple of length one is the linear derivative (in this case wrt to ctWRe). The quadratic derivatives are stored in the keys ('ctWRe', 'ctWRe') etc.
    # The list of Wilson coefficients is in: weightInfo.variables
    # The list of all derivatives (i.e., the list of all combiations of all variables up to length 2) is weightInfo.combinations. It includes the zeroth derivative, i.e., the constant term.

    # Let us scale all the weights to reasonable numbers. They come out of MG very small because the cross section of the process I used to genereate the top quarks is so small: s-channel single-top
    # Let us add up the constant terms of all events and normalize the sample to the number of events. (Arbitrary choice)
    const = (len(w[()])/w[()].sum())
    for k in w.keys():
        w[k] *= const 

    #let's remove the most extreme weight derivatives ... cosmetics for the propaganda plots
    from   tools import helpers 
    len_before = len(x)
    auto_clip = 0.001
    x, w = helpers.clip_quantile(x, auto_clip, weights = w )
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( auto_clip, len(x)/len_before ) )

    print ("Wilson coefficients:", weightInfo.variables )
    print ("Features of the first event:\n" + "\n".join( ["%25s = %4.3f"%(name, value) for name, value in zip(feature_names, x[0])] ) )
    prstr = {0:'constant', 1:'linear', 2:'quadratic'}
    print ("Weight coefficients(!) of the first event:\n"+"\n".join( ["%30s = %4.3E"%( prstr[len(comb)] + " " +",".join(comb), w[comb][0]) for comb in weightInfo.combinations] ) )

    # Let us compute the quadratic weight for ctWRe=1:
    import copy
    eft_sm  = make_eft()
    eft_bsm = make_eft(ctWRe=1)
    # constant term
    reweight = copy.deepcopy(w[()])
    # linear term
    for param1 in wilson_coefficients:
        reweight += (eft_bsm[param1]-eft_sm[param1])*w[(param1,)] 
    # quadratic term
    for param1 in wilson_coefficients:
        if eft_bsm[param1]-eft_sm[param1] ==0: continue
        for param2 in wilson_coefficients:
            if eft_bsm[param2]-eft_sm[param2] ==0: continue
            reweight += .5*(eft_bsm[param1]-eft_sm[param1])*(eft_bsm[param2]-eft_sm[param2])*w[tuple(sorted((param1,param2)))]

    print ("w(ctWRe=1) for the first event is: ", reweight[0])

    # Let us compute the weight ratio we can use for the training:
    target_ctWRe        = w[('ctWRe',)]/w[()]
    target_ctWRe_ctWRe  = w[('ctWRe','ctWRe')]/w[()]

    # NOTE!! These "target" branches are already written to the training data! No need to compute
