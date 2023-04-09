import pickle
import random
import ROOT
from math import pi
if __name__=="__main__":
    import sys
    sys.path.append('../../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

features = [   
    "parton_lepTop_pt", "parton_lepTop_phi", "parton_lepTop_lep_pt", "parton_lepTop_lep_eta", "parton_lepTop_lep_phi", "parton_lepTop_nu_pt", "parton_lepTop_nu_phi", "parton_lepTop_b_pt", "parton_lepTop_b_eta", "parton_lepTop_b_phi", "parton_lepTop_W_pt", "parton_lepTop_W_phi",
        "delphesJet_pt", "delphesJet_eta", "delphesJet_phi",   
        "delphesJet_SDmass", "delphesJet_SDsubjet0_deltaEta", "delphesJet_SDsubjet0_deltaPhi", "delphesJet_SDsubjet0_deltaR", "delphesJet_SDsubjet0_mass", "delphesJet_SDsubjet1_deltaEta",  "delphesJet_SDsubjet1_deltaPhi", "delphesJet_SDsubjet1_deltaR", "delphesJet_SDsubjet1_mass",  
        "delphesJet_tau1", "delphesJet_tau2", "delphesJet_tau3", "delphesJet_tau4", "delphesJet_tau21", "delphesJet_tau32", "delphesJet_ecf1", "delphesJet_ecf2", "delphesJet_ecf3", "delphesJet_ecfC1", "delphesJet_ecfC2", "delphesJet_ecfC3", "delphesJet_ecfD", "delphesJet_ecfDbeta2", "delphesJet_ecfM1", "delphesJet_ecfM2", "delphesJet_ecfM3", "delphesJet_ecfM1beta2", "delphesJet_ecfM2beta2", "delphesJet_ecfM3beta2", "delphesJet_ecfN1", "delphesJet_ecfN2", "delphesJet_ecfN3", "delphesJet_ecfN1beta2", "delphesJet_ecfN2beta2", "delphesJet_ecfN3beta2", "delphesJet_ecfU1", "delphesJet_ecfU2", "delphesJet_ecfU3", "delphesJet_ecfU1beta2", "delphesJet_ecfU2beta2", "delphesJet_ecfU3beta2",  
    ]

observers = ["parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k","parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", "parton_xi_nn", "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab"]

predictions =  ["ctGIm_lin_1_epoch_%i"%i for i in range(0,251,50)] #forgot to remove th lin from the name

data_generator =  DataGenerator(
    input_files = ["/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-lin/TT01j_HT800_ext_comb/output_*.root"],
        n_split = 1,
        splitting_strategy = "files",
        selection   = None,
        branches = ["p_C"] + features + observers + predictions) 

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

def getEvents( nTraining, return_observers = True):
    data_generator.load(-1, small=nTraining )
    combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch('p_C')
    if return_observers:
        return data_generator.scalar_branches( features ), {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}, data_generator.scalar_branches( observers )
    else:
        return data_generator.scalar_branches( features ), {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

tex = {"ctWRe":"C_{tW}^{Re}", "ctWIm":"C_{tW}^{Im}", "ctGRe":"C_{tG}^{Re}", "ctGIm":"C_{tG}^{Im}", "ctBIm":"C_{tB}^{Im}", "ctBRe":"C_{tB}^{Re}", "cHt":"C_{Ht}", 'cHtbRe':'C_{Htb}^{Re}', 'cHtbIm':'C_{Htb}^{Im}', 'cHQ3':'C_{HQ}^{(3)}'}

#['ctWRe', 'ctBRe', 'cHQ3', 'cHt', 'cHtbRe', 'ctWIm', 'ctBIm', 'cHtbIm']

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},

    {'color':ROOT.kMagenta-4,   'eft':make_eft(ctWRe=-1),  'tex':"Re(c_{tW})=-1", },
    {'color':ROOT.kMagenta+2,   'eft':make_eft(ctWRe=1),   'tex':"Re(c_{tW})=1",  },
    {'color':ROOT.kOrange-4,   'eft':make_eft(ctWIm=-1),  'tex':"Im(c_{tW})=-1", },
    {'color':ROOT.kOrange+2,   'eft':make_eft(ctWIm=1),   'tex':"Im(c_{tW})=1",  },
    {'color':ROOT.kGreen-4,     'eft':make_eft(ctGRe=-1),  'tex':"Re(c_{tG})=-1", },
    {'color':ROOT.kGreen+2,     'eft':make_eft(ctGRe=1),   'tex':"Re(c_{tG})=1",  },
    {'color':ROOT.kBlue-4,      'eft':make_eft(ctGIm=-1),  'tex':"Im(c_{tG})=-1", },
    {'color':ROOT.kBlue+2,      'eft':make_eft(ctGIm=1),   'tex':"Im(c_{tG})=1",  },
    ]

plot_options =  {
    "parton_hadTop_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t)'},
    "parton_hadTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t)'},
    "parton_hadTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t)'},
    "parton_lepTop_pt" :{'binning':[30,0,800], 'tex':'p_{T}(t lep)'},
    "parton_lepTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t lep)'},
    "parton_lepTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t lep)'},
    "parton_lepTop_lep_pt" :{'binning':[30,0,800], 'tex':'p_{T}(l (t lep))'},
    "parton_lepTop_lep_eta" :{'binning':[30,-3,3], 'tex':'#eta(l(t lep))'},
    "parton_lepTop_lep_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(l(t lep))'},
    "parton_lepTop_nu_pt" :{'binning':[30,0,800], 'tex':'p_{T}(#nu (t lep))'},
    "parton_lepTop_nu_eta" :{'binning':[30,-3,3], 'tex':'#eta(#nu(t lep))'},
    "parton_lepTop_nu_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(#nu(t lep))'},
    "parton_lepTop_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t lep))'},
    "parton_lepTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t lep))'},
    "parton_lepTop_b_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(b(t lep))'},
    "parton_lepTop_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t lep))'},
    "parton_lepTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t lep))'},
    "parton_lepTop_W_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(W(t lep))'},

    "delphesJet_pt"                 :{'binning':[50,500,2000], 'tex':'p_{T}(jet)'},
    "delphesJet_eta"                :{'binning':[30,-3,3], 'tex':'#eta(jet)'},
    "delphesJet_phi"                :{'binning':[30,-pi,pi], 'tex':'#phi(jet)'},
    "delphesJet_nConstituents"      :{'binning':[30,30,230], 'tex':'n-constituents'},
    "delphesJet_SDmass"             :{'binning':[30,150,200], 'tex':'M_{SD}(jet)'},
    "delphesJet_SDsubjet0_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{0})'},
    "delphesJet_SDsubjet1_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{1})'},
    "delphesJet_tau1"               :{'binning':[30,0,1], 'tex':'#tau_{1}'},
    "delphesJet_tau2"               :{'binning':[30,0,.5],'tex':'#tau_{2}'},
    "delphesJet_tau3"               :{'binning':[30,0,.3],'tex':'#tau_{3}'},
    "delphesJet_tau4"               :{'binning':[30,0,.3],'tex':'#tau_{4}'},
    "delphesJet_tau21"              :{'binning':[30,0,1], 'tex':'#tau_{21}'},
    "delphesJet_tau32"              :{'binning':[30,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "delphesJet_ecf1"               :{'binning':[30,0,2000], 'tex':"ecf1"},
    "delphesJet_ecf2"               :{'binning':[30,0,400000], 'tex':"ecf2"},
    "delphesJet_ecf3"               :{'binning':[30,0,4000000], 'tex':"ecf3"},
    "delphesJet_ecfC1"              :{'binning':[30,0,.5], 'tex':"ecfC1"},
    "delphesJet_ecfC2"              :{'binning':[30,0,.5], 'tex':"ecfC2"},
    "delphesJet_ecfC3"              :{'binning':[30,0,.5], 'tex':"ecfC3"},
    "delphesJet_ecfD"               :{'binning':[30,0,8], 'tex':"ecfD"},
    "delphesJet_ecfDbeta2"          :{'binning':[30,0,20], 'tex':"ecfDbeta2"},
    "delphesJet_ecfM1"              :{'binning':[30,0,0.35], 'tex':"ecfM1"},
    "delphesJet_ecfM2"              :{'binning':[30,0,0.2], 'tex':"ecfM2"},
    "delphesJet_ecfM3"              :{'binning':[30,0,0.2], 'tex':"ecfM3"},
    "delphesJet_ecfM1beta2"         :{'binning':[30,0,0.35], 'tex':"ecfM1beta2"},
    "delphesJet_ecfM2beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM2beta2"},
    "delphesJet_ecfM3beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM3beta2"},
    "delphesJet_ecfN1"              :{'binning':[30,0,0.5], 'tex':"ecfN1"},
    "delphesJet_ecfN2"              :{'binning':[30,0,0.5], 'tex':"ecfN2"},
    "delphesJet_ecfN3"              :{'binning':[30,0,5], 'tex':"ecfN3"},
    "delphesJet_ecfN1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN1beta2"},
    "delphesJet_ecfN2beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN2beta2"},
    "delphesJet_ecfN3beta2"         :{'binning':[30,0,5], 'tex':"ecfN3beta2"},
    "delphesJet_ecfU1"              :{'binning':[30,0,0.5], 'tex':"ecfU1"},
    "delphesJet_ecfU2"              :{'binning':[30,0,0.04], 'tex':"ecfU2"},
    "delphesJet_ecfU3"              :{'binning':[30,0,0.004], 'tex':"ecfU3"},
    "delphesJet_ecfU1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfU1beta2"},
    "delphesJet_ecfU2beta2"         :{'binning':[30,0,0.04], 'tex':"ecfU2beta2"},
    "delphesJet_ecfU3beta2"         :{'binning':[30,0,0.004], 'tex':"ecfU3beta2"},

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

if __name__=="__main__":
    import numpy as np
 
    # load some events and their weights 
    x,w,o = getEvents(-1, return_observers = True)

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

    auto_clip = 0.003
    quantiles = {feature: np.quantile( x[:, i_feature], ( auto_clip, 1.-auto_clip )) for i_feature, feature in enumerate(features)}
    print( "Clipping cut for %f"%auto_clip)
    print( " & ".join(["(%s>%f) & (%s<%f)"%( feature, quantiles[feature][0], feature, quantiles[feature][1]) for feature in features]) )
    
    #let's remove the most extreme weight derivatives ... cosmetics for the propaganda plots
    from   tools import helpers 
    len_before = len(x)
    x, w = helpers.clip_quantile(x, auto_clip, weights = w)
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( auto_clip, len(x)/len_before ) )

    print ("Wilson coefficients:", weightInfo.variables )
    print ("Features of the first event:\n" + "\n".join( ["%25s = %4.3f"%(name, value) for name, value in zip(features, x[0])] ) )
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
