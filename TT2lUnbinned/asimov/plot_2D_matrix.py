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

#dir = {'marginalized': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v2_2D/",
#       'marginalized_th_mod_exp': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v1_2D/"}
dir = {'marginalized': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v3_2D/",
       'marginalized_th_mod_exp': "/groups/hephy/cms/robert.schoefbeck/NN/results/TT2lUnbinned/limits/v3_2D/"}

# fake global object so pickle can unpickle. 
def model_weights():
    pass

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
subdirs = [ "marginalized", "marginalized_th_mod_exp"]
style = {0.68:ROOT.kDashed, 0.95:1}
color = {"marginalized": ROOT.kBlue, "marginalized_th_mod_exp":ROOT.kBlack }

#for wc1, wc2 in [('ctGIm', 'cQj18')]:#list( itertools.combinations( coefficients, 2)):
contours = {}
histo    = {}

for wc1, wc2 in list( itertools.combinations( coefficients, 2)):
    contours[(wc1, wc2)] = {}
    histo[(wc1, wc2)]    = {}

    for subdir in subdirs:
        result = getTGraphData( subdir,  wc1, wc2 )

        tgr = ROOT.TGraph2D( len(result), array.array('d', [r[1] for r in result]), array.array('d', [r[0] for r in result]), array.array('d', [r[2] for r in result]) )
        histo[(wc1, wc2)][subdir] = tgr.GetHistogram().Clone()
        contours[(wc1, wc2)][subdir] = {cl:cont for cl, cont in zip( confidence_levels, getContours( tgr.GetHistogram()) ) }
        histo[(wc1, wc2)][subdir].GetYaxis().SetRangeUser(*ranges[wc1])
        histo[(wc1, wc2)][subdir].GetXaxis().SetRangeUser(*ranges[wc2])

c1 = ROOT.TCanvas("","", 500*(len(coefficients)-1), 500*(len(coefficients)-1))
c1.Divide(len(coefficients)-1, len(coefficients)-1, 0.002, 0.002)
for wc1, wc2 in list( itertools.combinations( coefficients, 2)):
    c1.cd(coefficients.index(wc2) + (len(coefficients)-1)*coefficients.index(wc1))
    first = True
    for subdir in subdirs:

        h =  histo[(wc1, wc2)][subdir]

        for i_x in range(1, h.GetNbinsX()+1):
            for i_y in range(1, h.GetNbinsY()+1):
                h.SetBinContent( i_x, i_y, 0)
                h.SetBinError(   i_x, i_y, 0)

        h.GetYaxis().SetTitle(tex[wc1])
        h.GetXaxis().SetTitle(tex[wc2])

        h.GetYaxis().SetRangeUser(*ranges[wc1])
        h.GetXaxis().SetRangeUser(*ranges[wc2])

        h.GetZaxis().SetTitle("-2#Delta L");
        if first:
            h.Draw()
            first=False

        #c1.Range(ranges[wc1][0], ranges[wc2][0], ranges[wc1][1], ranges[wc2][1])
        for level, conts in contours[(wc1, wc2)][subdir].items():
            for c in conts:
                c.SetLineColor( color[subdir] )
                c.SetLineStyle( style[level] )
                c.Draw("lsame")
                print ("cx", h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())
                print ("cy", h.GetYaxis().GetXmin(), h.GetYaxis().GetXmax())

    #c1.Update()
    #c1.RedrawAxis()

out_dir = os.path.join( plot_directory, "limits_2D_3")
os.makedirs( out_dir, exist_ok=True)
c1.Print(os.path.join( out_dir, "matrix.png"))
c1.Print(os.path.join( out_dir, "matrix.pdf"))
copyIndexPHP( out_dir )

syncer.sync() 
