import glob
import os
import pickle
import Modeling
import ROOT
import array
import scipy

dir = "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v1_1D/"

# fake global object so pickle can unpickle. 
def model_weights():
    pass

configs = [
    {'color':ROOT.kGray, 'tex':'prefit',    'name':'def'},
    {'color':ROOT.kBlue, 'tex':'marg',          'name':'marginalized'},
    {'color':ROOT.kRed,  'tex':'marg th',       'name':'marginalized_th'},
    {'color':ROOT.kGreen,'tex':'marg th/exp',   'name':'marginalized_th_exp'},
    {'color':ROOT.kBlack,'tex':'marg th/mod/exp','name':'marginalized_th_mod_exp'},
    #{'color':ROOT.kMagenta,'tex':'marg th/mod',   'name':'marginalized_th_mod', 'style':ROOT.kDashed},
    #{'color':ROOT.k,'tex':'marg mod/exp',  'name':'marginalized_mod_exp'},
    {'color':ROOT.kGreen,'tex':'marg exp',      'name':'marginalized_exp', 'style':ROOT.kDashed},
    #{'color':ROOT.k,'tex':'th',            'name':'th'},
    #{'color':ROOT.k,'tex':'th/mod',        'name':'th_mod'},
    #{'color':ROOT.k,'tex':'th/mod/exp',    'name':'th_mod_exp'},
    #{'color':ROOT.k,'tex':'th/exp',        'name':'th_exp'},
    #{'color':ROOT.k,'tex':'exp',           'name':'exp'},
    #{'color':ROOT.k,'tex':'mod/exp',       'name':'mod_exp'},
]

def getTGraphData( subdir, wc):
    files = glob.glob(os.path.join( dir, subdir )+'/%s*.pkl'%wc)

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
        median_ll = scipy.stats.ncx2.median(df=1,nc=nc)
        if median_ll>-float('inf'):
            results.append( (res[hypothesis][wc].val, median_ll ) ) 

    results.sort()
    return results

        #Modeling.median_expected_pValue( df=1,

stuff = []

for x_range, wc in [
        ((-0.8, 0.8), 'ctGRe'), 
        ((-0.8, 0.8), 'ctGIm'), 
        ((-2.0, 1.2), 'cQj18'), 
        ((-2.0, 2.0), 'cQj38'), 
        ((-1.2, 0.8), 'ctj8')
        ]:

    c1 = ROOT.TCanvas()
    first = True
    for cfg in reversed(configs):
        result = getTGraphData( cfg['name'],  wc )

        tgr = ROOT.TGraph( len(result), array.array('d', [r[0] for r in result]), array.array('d', [r[1] for r in result]) )
        stuff.append( tgr )

        tgr.SetLineColor(cfg['color']);
        tgr.SetLineWidth(2);
        if 'style' in cfg:
            tgr.SetLineStyle(cfg['style']);

        tgr.SetMarkerColor(cfg['color']);
        tgr.SetMarkerSize(0);
        tgr.SetMarkerStyle(0);
        tgr.GetXaxis().SetTitle(wc);
        tgr.GetYaxis().SetTitle("-2#Delta L");

        if first:
            tgr.Draw("ALP")
            first=False
        else:
            tgr.Draw("LP")
        tgr.GetXaxis().SetRangeUser(*x_range) 
        tgr.GetYaxis().SetRangeUser(0,25) 
    c1.Print(wc+".png")    
