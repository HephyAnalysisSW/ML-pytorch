
# Standard imports
import ROOT
import numpy as np
import os, sys
sys.path.insert(0, '..')

# RootTools
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers

import tools.syncer as syncer

#    c1 = ROOT.TCanvas("c1");
#
#    l = ROOT.TLegend(0.2,0.8,0.9,0.85)
#    l.SetNColumns(2)
#    l.SetFillStyle(0)
#    l.SetShadowColor(ROOT.kWhite)
#    l.SetBorderSize(0)
#
#    # GIF animation
#    tex = ROOT.TLatex()
#    tex.SetNDC()
#    tex.SetTextSize(0.06)

def training_plot( model, plot_directory, training_features, training_weights, test_features, test_weights, label = None):
    stuff = []

    # colors
    color = {}
    i_lin, i_diag, i_mixed = 0,0,0
    derivatives = list( [ k for k in training_weights.keys() if k is not tuple() ] )
    for i_der, der in enumerate(derivatives):
        if len(der)==1:
            color[der] = ROOT.kAzure + i_lin
            i_lin+=1
        elif len(der)==2 and len(set(der))==1:
            color[der] = ROOT.kRed + i_diag
            i_diag+=1
        elif len(der)==2 and len(set(der))==2:
            color[der] = ROOT.kGreen + i_mixed
            i_mixed+=1

    w0 = test_weights[()]
    h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
    for i_feature, feature in enumerate(model.feature_names):
        # root style binning
        binning     = model.plot_options[feature]['binning']
        # linspace binning
        lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
        # digitize feature
        binned      = np.digitize(test_features[:,i_feature], lin_binning[feature] )
        # for each digit, create a mask to select the corresponding event in the bin (e.g. test_features[mask[0]] selects features in the first bin
        mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )

        h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
        h_derivative_prediction = np.array([ (w0.reshape(-1,1)*test_predictions)[m].sum(axis=0) for m in mask])
        h_derivative_truth      = np.array([ (np.transpose(np.array([test_weights[der] for der in derivatives])))[m].sum(axis=0) for m in mask])

        h_ratio_prediction[feature] = h_derivative_prediction/h_w0[feature].reshape(-1,1) 
        h_ratio_truth[feature]      = h_derivative_truth/h_w0[feature].reshape(-1,1)

    n_pads = len(model.feature_names)+1
    n_col  = min(4, n_pads)
    n_rows = n_pads//n_col
    if n_rows*n_col<n_pads: n_rows+=1

    for logY in [False, True]:
        c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
        c1.Divide(n_col,n_rows)

        l = ROOT.TLegend(0.2,0.1,0.9,0.85)
        stuff.append(l)
        l.SetNColumns(2)
        l.SetFillStyle(0)
        l.SetShadowColor(ROOT.kWhite)
        l.SetBorderSize(0)

        for i_feature, feature in enumerate(model.feature_names):

            th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
            c1.cd(i_feature+1)
            ROOT.gStyle.SetOptStat(0)
            th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( derivatives ) }
            th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( derivatives ) }

            stuff.append(th1d_yield)
            stuff.append(th1d_ratio_truth)
            stuff.append(th1d_ratio_pred)

            th1d_yield.SetLineColor(ROOT.kGray+2)
            th1d_yield.SetMarkerColor(ROOT.kGray+2)
            th1d_yield.SetMarkerStyle(0)
            th1d_yield.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
            th1d_yield.SetTitle("")

            th1d_yield.Draw("hist")

            for i_der, der in enumerate(derivatives):
                th1d_ratio_truth[der].SetTitle("")
                th1d_ratio_truth[der].SetLineColor(color[der])
                th1d_ratio_truth[der].SetMarkerColor(color[der])
                th1d_ratio_truth[der].SetMarkerStyle(0)
                th1d_ratio_truth[der].SetLineWidth(2)
                th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                th1d_ratio_truth[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                th1d_ratio_pred[der].SetTitle("")
                th1d_ratio_pred[der].SetLineColor(color[der])
                th1d_ratio_pred[der].SetMarkerColor(color[der])
                th1d_ratio_pred[der].SetMarkerStyle(0)
                th1d_ratio_pred[der].SetLineWidth(2)
                th1d_ratio_pred[der].GetXaxis().SetTitle(model.plot_options[feature]['tex'])

                tex_name = "_{%s}"%(",".join([model.tex[c] for c in der]))

                if i_feature==0:
                    l.AddEntry( th1d_ratio_truth[der], "R"+tex_name)
                    l.AddEntry( th1d_ratio_pred[der],  "#hat{R}"+tex_name)

            if i_feature==0:
                l.AddEntry( th1d_yield, "yield (SM)")

            max_ = max( map( lambda h:h.GetMaximum(), th1d_ratio_truth.values() ))
            max_ = 10**(1.5)*max_ if logY else 1.5*max_
            min_ = min( map( lambda h:h.GetMinimum(), th1d_ratio_truth.values() ))
            min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

            th1d_yield_min = th1d_yield.GetMinimum()
            th1d_yield_max = th1d_yield.GetMaximum()
            for bin_ in range(1, th1d_yield.GetNbinsX() ):
                th1d_yield.SetBinContent( bin_, (th1d_yield.GetBinContent( bin_ ) - th1d_yield_min)/th1d_yield_max*(max_-min_)*0.95 + min_  )

            #th1d_yield.Scale(max_/th1d_yield.GetMaximum())
            th1d_yield   .Draw("hist")
            ROOT.gPad.SetLogy(logY)
            th1d_yield   .GetYaxis().SetRangeUser(min_, max_)
            th1d_yield   .Draw("hist")
            for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                h .Draw("hsame")

        c1.cd(len(model.feature_names)+1)
        l.Draw()

        lines = [ (0.29, 0.9, 'N_{B} =%5i'%( max_n_tree )) ]
        drawObjects = [ tex.DrawLatex(*line) for line in lines ]
        for o in drawObjects:
            o.Draw()

        plot_directory_ = os.path.join( plot_directory, "log" if logY else "lin" )
        if not os.path.isdir(plot_directory_):
            try:
                os.makedirs( plot_directory_ )
            except IOError:
                pass
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, "training_%s.png"%(label) ) )
        syncer.makeRemoteGif(plot_directory_, pattern="training_*.png", name="training" )
    syncer.sync()
