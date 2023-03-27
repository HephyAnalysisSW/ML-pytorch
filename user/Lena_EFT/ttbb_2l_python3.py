#!/usr/bin/env python

#Standard imports
from operator                   import attrgetter
from math                       import pi, sqrt, cosh, cos, acos, sin, sinh
#import ROOT, os
import copy
import itertools

#From RootTools
# from RootTools.core.standard     import *

#From Tools
#from tttt.Tools.helpers          import deltaPhi, deltaR2, deltaR, getCollection, getObjDict
#from Analysis.Tools.WeightInfo       import WeightInfo

import logging
logger = logging.getLogger(__name__)


#from Analysis.Tools.mt2Calculator import mt2Calculator
#mt2Calculator = mt2Calculator()

# Read variables and sequences
jetVars          = ['pt/F', 'eta/F', 'phi/F', 'bTag/F', 'bTagPhys/I']
jetVarNames      = [x.split('/')[0] for x in jetVars]

lepVars          = ['pt/F','eta/F','phi/F','pdgId/I','isolationVar/F', 'isolationVarRhoCorr/F']
lepVarNames      = [x.split('/')[0] for x in lepVars]

lstm_jets_maxN   = 10
lstm_jetVars     = ['pt/F', 'eta/F', 'phi/F', 'bTag/F']
lstm_jetVarNames = [x.split('/')[0] for x in lstm_jetVars]

# Training variables
read_variables = [\
    "nBTag/I", 
    "recoMet_pt/F", "recoMet_phi/F",
    "genMet_pt/F", "genMet_phi/F",
    "nrecoJet/I",
    "recoJet[%s]"%(",".join(jetVars)),
    "nrecoLep/I",
    "recoLep[%s]"%(",".join(lepVars)),
    "ngenLep/I", "genLep[pt/F,eta/F,phi/F,pdgId/I,mother_pdgId/I]", 
    "lumiweight1fb/F",
    "evt/l", "run/I", "lumi/I",
    "recoBj0_pt/F",
    "recoBj1_pt/F",
]

plot_mva_variables = {
    "mT_l1"                 : [[0, 800 ], '$ m_{T}(l_{1}) $'] ,  
    "l1_pt"                 : [[0, 300 ], '$ p_{T}(l_{1})  $'] , 
    "l1_eta"                : [[-3, 3  ], '$ \eta(l_{1}) $'] ,  
    "met_pt"                : [[0, 400 ], '$ E_{T}^{miss}  $'] ,  
    "mj_12"                 : [[0, 2500], '$ m_{2j} $'] ,  
    "mlj_11"                : [[0, 2500], '$ m_{l1,j1} $'] ,     
    "dPhil_12"              : [[0, 3.5 ], '$ \Delta\phi_{2l} $'] ,  
    "dPhij_12"              : [[0, 3.5 ], '$ \Delta\phi_{2j} $'] ,  
    "dEtal_12"              : [[0, 6   ], '$ \Delta\eta_{2l} $'] ,  
    "dEtaj_12"              : [[0, 6   ], '$ \Delta\eta_{2j} $'] ,  
    "ht"                    : [[0, 3000], '$ H_{T} $'] ,  
    "htb"                   : [[0, 2500], '$ H_{T,b-jets} $'] ,  
    "jet0_pt"               : [[0, 600 ], '$ p_{T}(j_{0})  $'] ,  
    "jet0_eta"              : [[-3, 3  ], '$ \eta(j_{0}) $'] ,  
    "bjet0_pt"              : [[0, 600 ], '$ p_{T}(b_{0})  $'] ,  
    "bjet1_pt"              : [[0, 600 ], '$ p_{T}(b_{1})  $'] , 
    "m_4b"                  : [[0, 2500], '$ m_{4b}  $'] ,  
    "mt2bb"                 : [[0, 1200], '$ m2_{T,bb} $'] ,  
    "min_dR_bb"             : [[0, 3.5 ], '$ \Delta R_{b-jet,b-jet} $'] ,  
    "dR_2l"                 : [[0, 3.5 ], '$ \Delta R_{2l} $'] ,  
    # "nrecoJet"              : [[0, 8   ], '$ jet multiplicity $'] ,  
    # "mT_l2"                 : [[0, 800 ], '$ m_{T}(l_{2}) $'] ,  
    # "ml_12"                 : [[0, 1500], '$ m_{2l} $'] ,  
    # "l2_pt"                 : [[0, 300 ], '$ p_{T}(l_{2})  $'] ,  
    # "l2_eta"                : [[-3, 3  ], '$ \eta(l_{2}) $'] ,  
    # "mlj_12"                : [[0, 2500], '$ m_{l1, j2} $'] ,  
    # "ht_ratio"              : [[0, 1   ], '$ \Delta H_{T} $'] ,  
    # "jet1_pt"               : [[0, 600 ], '$ p_{T}(j_{1})  $'] ,  
    # "jet1_eta"              : [[-3, 3  ], '$ \eta(j_{1}) $'] ,  
    # "jet2_pt"               : [[0, 600 ], '$ p_{T}(j_{2})  $'] ,  
    # "jet2_eta"              : [[-3, 3  ], '$ \eta(j_{2}) $'] ,  
    # "jet3_pt"               : [[0, 600 ], '$ p_{T}(j_{3})  $'] ,  
    # "jet4_pt"               : [[0, 600 ], '$ p_{T}(j_{4})  $'] ,  
    # "jet5_pt"               : [[0, 600 ], '$ p_{T}(j_{5})  $'] ,  
    # "jet6_pt"               : [[0, 600 ], '$ p_{T}(j_{6})  $'] ,  
    # "jet7_pt"               : [[0, 600 ], '$ p_{T}(j_{7})  $'] ,  
    # "dR_min0"               : [[0, 3.5 ], '$ \Delta R_{0} $'] ,  
    # "dR_min1"               : [[0, 3.5 ], '$ \Delta R_{1} $'] ,  
    # "mt2ll"                 : [[0, 1200], '$ m2_{T,ll} $'] ,  
    # "mt2blbl"               : [[0, 1200], '$ m2_{T,blbl} $'] ,  
    }
    
    
    
    


all_mva_variables = {

     #Global Event Properties
     "nrecoJet"              :(lambda event, sample: event.nrecoJet),

     #Lep Vars
     "mT_l1"                 :(lambda event, sample: sqrt(2*event.recoLep_pt[0]*event.recoMet_pt*(1-cos(event.recoLep_phi[0]-event.recoMet_phi)))),
     "mT_l2"                 :(lambda event, sample: sqrt(2*event.recoLep_pt[1]*event.recoMet_pt*(1-cos(event.recoLep_phi[1]-event.recoMet_phi)))),
     "ml_12"                 :(lambda event, sample: sqrt(2*event.recoLep_pt[0]*event.recoLep_pt[1]*(cosh(event.recoLep_eta[0]-event.recoLep_eta[1])-cos(event.recoLep_phi[0]-event.recoLep_phi[1])))),
     "l1_pt"                 :(lambda event, sample: event.recoLep_pt[0]),
     "l1_eta"                :(lambda event, sample: event.recoLep_eta[0]),
     "l2_pt"                 :(lambda event, sample: event.recoLep_pt[1]),
     "l2_eta"                :(lambda event, sample: event.recoLep_eta[1]),

     #MET
     "met_pt"                :(lambda event, sample: event.recoMet_pt),
     
     #leptons/jets
     "mj_12"                 :(lambda event, sample: sqrt(2*event.recoJet_pt[0]*event.recoJet_pt[1]*(cosh(event.recoJet_eta[0]-event.recoJet_eta[1])-cos(event.recoJet_phi[0]-event.recoJet_phi[1]))) if event.nrecoJet >=2 else 0),
     "mlj_11"                :(lambda event, sample: sqrt(2*event.recoLep_pt[0]*event.recoJet_pt[0]*(cosh(event.recoLep_eta[0]-event.recoJet_eta[0])-cos(event.recoLep_phi[0]-event.recoJet_phi[0]))) if event.nrecoJet >=1 else 0),
     "mlj_12"                :(lambda event, sample: sqrt(2*event.recoLep_pt[0]*event.recoJet_pt[1]*(cosh(event.recoLep_eta[0]-event.recoJet_eta[1])-cos(event.recoLep_phi[0]-event.recoJet_phi[1]))) if event.nrecoJet >=2 else 0),
     
     "dPhil_12"              :(lambda event, sample: deltaPhi(event.recoLep_phi[0], event.recoLep_phi[1])),
     "dPhij_12"              :(lambda event, sample: deltaPhi(event.recoJet_phi[0], event.recoJet_phi[1]) if event.nrecoJet >=2 else 0),
     
     "dEtal_12"              :(lambda event, sample: abs(event.recoLep_eta[0]-event.recoLep_eta[1])),
     "dEtaj_12"              :(lambda event, sample: abs(event.recoJet_eta[0]-event.recoJet_eta[1]) if event.nrecoJet >=2 else -10),

     "ht"                    :(lambda event, sample: sum( [event.recoJet_pt[i] for i in range (event.nrecoJet)] ) ),
     "htb"                   :(lambda event, sample: sum( [j['pt'] for j in event.recoBJets] ) ),
     "ht_ratio"              :(lambda event, sample: sum( [event.recoJet_pt[i] for i in range (4)])/ sum( [event.recoJet_pt[i] for i in range (event.nrecoJet) ]) if event.nrecoJet>=4 else 1 ),

     "jet0_pt"               :(lambda event, sample: event.recoJet_pt[0]          if event.nrecoJet >=1 else 0),
     "jet0_eta"              :(lambda event, sample: event.recoJet_eta[0]         if event.nrecoJet >=1 else -10),
     "jet1_pt"               :(lambda event, sample: event.recoJet_pt[1]          if event.nrecoJet >=2 else 0),
     "jet1_eta"              :(lambda event, sample: event.recoJet_eta[1]         if event.nrecoJet >=2 else -10),
     "jet2_pt"               :(lambda event, sample: event.recoJet_pt[2]          if event.nrecoJet >=3 else 0),
     "jet2_eta"              :(lambda event, sample: event.recoJet_eta[2]         if event.nrecoJet >=3 else -10),
     "jet3_pt"               :(lambda event, sample: event.recoJet_pt[3]          if event.nrecoJet >=4 else 0),
     "jet4_pt"               :(lambda event, sample: event.recoJet_pt[4]          if event.nrecoJet >=5 else 0),
     "jet5_pt"               :(lambda event, sample: event.recoJet_pt[5]          if event.nrecoJet >=6 else 0),
     "jet6_pt"               :(lambda event, sample: event.recoJet_pt[6]          if event.nrecoJet >=7 else 0),
     "jet7_pt"               :(lambda event, sample: event.recoJet_pt[7]          if event.nrecoJet >=8 else 0),
     "bjet0_pt"              :(lambda event, sample: event.recoBj0_pt),
     "bjet1_pt"              :(lambda event, sample: event.recoBj1_pt),
     "m_4b"                  :(lambda event, sample: event.m_4b),
     "dR_min0"               :(lambda event, sample: event.dR_min0),
     "dR_min1"               :(lambda event, sample: event.dR_min1),
     #Smallest dR between two b-tagged jets
     "min_dR_bb"             :(lambda event, sample: event.min_dR_bb),
     #dR between leading/subleading lepton
     "dR_2l"                 :(lambda event, sample: sqrt(deltaPhi(event.recoLep_phi[0], event.recoLep_phi[1])**2 + (event.recoLep_eta[0] - event.recoLep_eta[1])**2)),
     #Mt2 mass lepton + b-jet
     "mt2ll"                 :(lambda event, sample : event.mt2ll),
     "mt2bb"                 :(lambda event, sample : event.mt2bb),
     "mt2blbl"               :(lambda event, sample : event.mt2blbl),
     }


## Using all variables
mva_variables_ = dict(sorted(all_mva_variables.items()))
mva_variables  = [ (key, value) for key, value in all_mva_variables.items() if key in mva_variables_ ]

def lstm_jets(event, sample):
    #print event.recoJet_bTagPhys[0]
    jets = [ getObjDict( event, 'recoJet_', lstm_jetVarNames, i) for i in range(int(event.nrecoJet)) ]
    #jets = filter( jet_vector_var['selector'], jets )
    return jets

# for the filler
mva_vector_variables    =   {
    #"mva_Jet":  {"name":"Jet", "vars":lstm_jetVars, "varnames":lstm_jetVarNames, "selector": (lambda jet: True), 'maxN':10}
    "mva_Jet":  {"func":lstm_jets, "name":"Jet", "vars":lstm_jetVars, "varnames":lstm_jetVarNames}
}


sequence = []

def addTLorentzVector( p_dict , name ):
    ''' add a TLorentz 4D Vector for further calculations
    '''
    #v = ROOT.TLorentzVector(0,0,0,0)
    p_dict[name] = ROOT.TLorentzVector( p_dict['pt']*cos(p_dict['phi']), p_dict['pt']*sin(p_dict['phi']),  p_dict['pt']*sinh(p_dict['eta']), p_dict['pt']*cosh(p_dict['eta']) )


def make_bjets ( event, sample ):

    event.recoJets = getCollection( event, 'recoJet', ['pt', 'eta', 'phi', 'bTag_loose'], 'nrecoJet' )
    event.recoBJets    = filter( lambda j:     j['bTag_loose'] and abs(j['eta'])<2.4 , event.recoJets )
    event.recoNonBJets = []
    for b in event.recoJets:
        addTLorentzVector( b, 'vec4D' ) 
        if b not in event.recoBJets:
            event.recoNonBJets.append(b)
    if (event.nrecoJet >= 4):        
        event.m_4b  = abs((event.recoJets[0]['vec4D']+event.recoJets[1]['vec4D']+event.recoJets[2]['vec4D']+event.recoJets[3]['vec4D']).M())  
    else:
        event.m_4b  = 0
    # minDR of all btag combinations
    if len(event.recoBJets)>=2:
        event.min_dR_bb = min( [deltaR( comb[0], comb[1] ) for comb in itertools.combinations( event.recoBJets, 2)] )
    else:
        event.min_dR_bb = -1
        
sequence.append ( make_bjets )

def make_leptons(event, sample):
    event.leptons   = getCollection(event, 'recoLep', lepVarNames, 'nrecoLep') 
    
     #(Second) smallest dR between any lepton and medium b-tagged jet
    dR_vals = sorted([deltaR(event.recoBJets[i], event.leptons[j]) for i in range(len(event.recoBJets)) for j in range(len(event.leptons))])
    if len(dR_vals)>=2:
        event.dR_min0 = dR_vals[0]
        event.dR_min1 = dR_vals[1]
    else:
        event.dR_min0 = -1
        event.dR_min1 = -1

sequence.append(make_leptons)




def MT2(event, sample):

    # we must always have two leptons and two jets, hence no default needed.
    mt2Calculator.reset()    
    mt2Calculator.setLeptons(event.recoLep_pt[0], event.recoLep_eta[0], event.recoLep_phi[0],event.recoLep_pt[1], event.recoLep_eta[1], event.recoLep_phi[1])
    mt2Calculator.setMet(event.recoMet_pt, event.recoMet_phi)
    event.mt2ll = mt2Calculator.mt2ll()
    
    
    b = (event.recoBJets + event.recoNonBJets )[:2]
    b1, b2 = b[0], b[1]
    mt2Calculator.setBJets(b1['pt'], b1['eta'], b1['phi'], b2['pt'], b2['eta'], b2['phi'])
    event.mt2bb = mt2Calculator.mt2bb()
    event.mt2blbl = mt2Calculator.mt2blbl()
    

sequence.append( MT2 )


total_genWeight = {
"TTTT_MS"   :3.07644844055,
"TT_2L"     :10325881856.0,
"TTbb_MS"   :258.110778809,
}

xsec = {
"TTTT_MS"   :11.9e-03,
"TT_2L"     :831.762*((3*0.108)**2),
"TTbb_MS"   :5.253e-01,
}
