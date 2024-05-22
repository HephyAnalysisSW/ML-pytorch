import glob
import os
import pickle
import Modeling
import ROOT
import array
import scipy
import itertools

from tools.user import plot_directory
import tools.syncer as syncer
from tools.helpers import copyIndexPHP

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

dir = {'marginalized': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v2_2D/",
       'marginalized_th_mod_exp': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v1_2D/"}

# fake global object so pickle can unpickle. 
def model_weights():
    pass

##config = marginalized 
##[
#    #{'color':ROOT.kGray, 'tex':'prefit',    'name':'def'},
#    {'color':ROOT.kBlue, 'tex':'marg',          'name':'marginalized'},
#    {'color':ROOT.kRed,  'tex':'marg th',       'name':'marginalized_th'},
#    {'color':ROOT.kGreen,'tex':'marg th/exp',   'name':'marginalized_th_exp'},
#    {'color':ROOT.kBlack,'tex':'marg th/mod/exp','name':'marginalized_th_mod_exp'},
#    #{'color':ROOT.kMagenta,'tex':'marg th/mod',   'name':'marginalized_th_mod', 'style':ROOT.kDashed},
#    #{'color':ROOT.k,'tex':'marg mod/exp',  'name':'marginalized_mod_exp'},
#    {'color':ROOT.kGreen,'tex':'marg exp',      'name':'marginalized_exp', 'style':ROOT.kDashed},
#    #{'color':ROOT.k,'tex':'th',            'name':'th'},
#    #{'color':ROOT.k,'tex':'th/mod',        'name':'th_mod'},
#    #{'color':ROOT.k,'tex':'th/mod/exp',    'name':'th_mod_exp'},
#    #{'color':ROOT.k,'tex':'th/exp',        'name':'th_exp'},
#    #{'color':ROOT.k,'tex':'exp',           'name':'exp'},
#    #{'color':ROOT.k,'tex':'mod/exp',       'name':'mod_exp'},
#]


def getTGraphData( subdir, wc1, wc2):
    files = glob.glob(os.path.join( dir[subdir], subdir )+'/%s_*_%s_*.pkl'%(wc1, wc2))
    files+= glob.glob(os.path.join( dir[subdir], subdir )+'/%s_*_%s_*.pkl'%(wc2, wc1))

    results = []
    for file in files:
        res = pickle.load(open(file,'rb'))
        if 'def' not in subdir:
            nonCentrality, hypothesis = 'nonCentrality', 'hypothesis'
        else:
            nonCentrality, hypothesis = 'preFit_nonCentrality', 'preFit_hypothesis'
        nc  = res[nonCentrality]
        if hypothesis not in res:
            print("No %s for %s"%(hypothesis, file))
            continue
        median_ll = scipy.stats.ncx2.median(df=2,nc=nc)
        if median_ll>-float('inf'):
            results.append( (res[hypothesis][wc1].val, res[hypothesis][wc2].val, median_ll ) ) 

    results.sort()
    return results

        #Modeling.median_expected_pValue( df=1,

confidence_levels = [0.68, 0.95]

def_contlist = [ scipy.stats.chi2.isf(1-cl, df=2) for cl in confidence_levels ]

def getContours(h, contlist=def_contlist):
    import ctypes
    _h = h.Clone()
    c_contlist = ((ctypes.c_double)*(len(contlist)))(*contlist)
    ctmp = ROOT.TCanvas()
    _h.SetContour(len(contlist),c_contlist)
    _h.Draw("contzlist")
    #_h.Draw("COL")
    #_h.GetZaxis().SetRangeUser(0.01,3)
    ctmp.Update()
    contours = ROOT.gROOT.GetListOfSpecials().FindObject("contours")

    contours_ = []
    for idx in range( len(contlist)):
        graph_list = contours.At(idx)
        contours_.append([])
        #print contours, ROOT.gROOT.GetListOfSpecials(), graph_list.GetEntries()
        for i in range(graph_list.GetEntries()):
                contours_[idx].append( graph_list.At(i).Clone("cont_"+str(i)) )

    return contours_

ranges = { 
    'ctGRe':(-0.6, 0.6),
    'ctGIm':(-0.6, 0.6),
    'cQj18':(-2.0, 1.2),
    'cQj38':(-2.0, 2.0),
    'ctj8':(-1.2, 0.8),
    }

coefficients = ['ctGRe', 'ctGIm', 'cQj18', 'cQj38', 'ctj8']
tex  = {'ctGRe':'C_{tG}^{Re}', 'ctGIm':'C_{tG}^{Im}', 'cQj18':'C_{Qj}^{18}', 'cQj38':'C_{Qj}^{38}', 'ctj8':'C_{tj}^{8}'}

style = {0.68:ROOT.kDashed, 0.95:1}
color = {"marginalized": ROOT.kBlue, "marginalized_th_mod_exp":ROOT.kBlack }

stuff = []
#for wc1, wc2 in [('ctGIm', 'cQj18')]:#list( itertools.combinations( coefficients, 2)):
for wc1, wc2 in list( itertools.combinations( coefficients, 2)):

    contours = {}
    histo    = {}
    for subdir in [ "marginalized", "marginalized_th_mod_exp"]:
        result = getTGraphData( subdir,  wc1, wc2 )

        tgr = ROOT.TGraph2D( len(result), array.array('d', [r[0] for r in result]), array.array('d', [r[1] for r in result]), array.array('d', [r[2] for r in result]) )
        histo[subdir] = tgr.GetHistogram().Clone()
        contours[subdir] = {cl:cont for cl, cont in zip( confidence_levels, getContours( tgr.GetHistogram()) ) }
        histo[subdir].GetXaxis().SetRangeUser(*ranges[wc1])
        histo[subdir].GetYaxis().SetRangeUser(*ranges[wc2])
        stuff.append( histo[subdir] )


    c1 = ROOT.TCanvas()
    c1.cd()
    first = True
    for subdir in [ "marginalized_th_mod_exp", "marginalized" ]:

        h =  histo[subdir]

        for i_x in range(1, h.GetNbinsX()+1):
            for i_y in range(1, h.GetNbinsY()+1):
                h.SetBinContent( i_x, i_y, 0)
                h.SetBinError(   i_x, i_y, 0)

        h.GetXaxis().SetTitle(tex[wc1])
        h.GetYaxis().SetTitle(tex[wc2])

        h.GetZaxis().SetTitle("-2#Delta L");
        if first:
            h.Draw("COLZ")
            first=False
    
        for level, conts in contours[subdir].items():
            for c in conts:
                c.SetLineColor( color[subdir] )
                c.SetLineStyle( style[level] )
                c.Draw("same")
                stuff.append( c )

    #c1.Update()
    #c1.RedrawAxis()

    out_dir = os.path.join( plot_directory, "limits_2D_2")
    os.makedirs( out_dir, exist_ok=True)
    c1.Print(os.path.join( out_dir, wc1+'_'+wc2+"_"+subdir+".png"))
    c1.Print(os.path.join( out_dir, wc1+'_'+wc2+"_"+subdir+".pdf"))
    copyIndexPHP( out_dir )

syncer.sync() 
