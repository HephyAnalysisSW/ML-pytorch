#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import os, sys, copy
sys.path.insert(0, '../../..')
from math import sqrt

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers
import tools.syncer as syncer

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="HadronicSMEFT",                 help="plot sub-directory")
argParser.add_argument("--model",              action="store",      default="ctGIm_lin_1",                  help="Which model?")
argParser.add_argument("--prefix",             action="store",      default="v2", type=str,  help="prefix")
argParser.add_argument('--feature_plots',      action='store_true', help="Feature plots?")
argParser.add_argument("--WC",                 action="store",      default="ctGIm", type=str,  help="Which WC?")
argParser.add_argument("--epochs",             action="store",      nargs="*", type=int,  help="Which epochs to plot?")
argParser.add_argument("--input_files",        action="store",      default="/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/TT01j_HT800_ext_comb/output_*.root", type=str,  help="input files")

args = argParser.parse_args()

#exec('import %s as model'%(args.model))
import particleNet_data as model 

features = model.features

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix if args.prefix is not None else "", args.model )
os.makedirs( plot_directory, exist_ok=True)

features, weights, observers = model.getEvents(-1, return_observers = True)

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

###############
## Plot Model #
###############

stuff = []
if args.feature_plots and hasattr( model, "eft_plot_points" ):
    h    = {}
    h_lin= {}
    for i_eft, eft_plot_point in enumerate(model.eft_plot_points):
        eft = eft_plot_point['eft']

        if i_eft == 0:
            eft_sm     = eft

        name = ''
        name= '_'.join( [ (wc+'_%3.2f'%eft[wc]).replace('.','p').replace('-','m') for wc in model.wilson_coefficients if wc in eft ])
        tex_name = eft_plot_point['tex'] 

        if i_eft==0: name='SM'

        h[name]     = {}
        h_lin[name] = {}

        eft['name'] = name
        
        for i_feature, feature in enumerate(features):
            h[name][feature]        = ROOT.TH1F(name+'_'+feature+'_nom',    name+'_'+feature, *model.plot_options[feature]['binning'] )
            h_lin[name][feature]    = ROOT.TH1F(name+'_'+feature+'_nom_lin',name+'_'+feature+'_lin', *model.plot_options[feature]['binning'] )

        # make reweights for x-check
        reweight     = copy.deepcopy(weights[()])
        # linear term
        for param1 in model.wilson_coefficients:
            reweight += (eft[param1]-eft_sm[param1])*weights[(param1,)] 
        reweight_lin  = copy.deepcopy( reweight )
        # quadratic term
        for param1 in model.wilson_coefficients:
            if eft[param1]-eft_sm[param1] ==0: continue
            for param2 in model.wilson_coefficients:
                if eft[param2]-eft_sm[param2] ==0: continue
                reweight += (.5 if param1!=param2 else 1)*(eft[param1]-eft_sm[param1])*(eft[param2]-eft_sm[param2])*weights[tuple(sorted((param1,param2)))]

        sign_postfix = ""

        for i_feature, feature in enumerate(features):
            binning = model.plot_options[feature]['binning']

            h[name][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight) )
            h_lin[name][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=reweight_lin) )

            h[name][feature].SetLineWidth(2)
            h[name][feature].SetLineColor( eft_plot_point['color'] )
            h[name][feature].SetMarkerStyle(0)
            h[name][feature].SetMarkerColor(eft_plot_point['color'])
            h[name][feature].legendText = tex_name
            h_lin[name][feature].SetLineWidth(2)
            h_lin[name][feature].SetLineColor( eft_plot_point['color'] )
            h_lin[name][feature].SetMarkerStyle(0)
            h_lin[name][feature].SetMarkerColor(eft_plot_point['color'])
            h_lin[name][feature].legendText = tex_name+(" (lin)" if name!="SM" else "")

    for i_feature, feature in enumerate(features):

        for _h in [h, h_lin]:
            norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
            if norm>0:
                for eft_plot_point in model.eft_plot_points:
                    _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 

        for postfix, _h in [ ("", h), ("_linEFT", h_lin)]:
            histos = [_h[eft_plot_point['eft']['name']][feature] for eft_plot_point in model.eft_plot_points]
            max_   = max( map( lambda h__:h__.GetMaximum(), histos ))

            for logY in [True, False]:

                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.68,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "feature_plots", "log" if logY else "lin" )
                assert False, ""
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))

            # Norm all shapes to 1
            for i_histo, histo in enumerate(histos):
                norm = histo.Integral()
                if norm>0:
                    histo.Scale(1./histo.Integral())

            # Divide all shapes by the SM
            ref = histos[0].Clone()
            for i_histo, histo in enumerate(histos):
                histo.Divide(ref)

            # Now plot shape differences
            for logY in [True, False]:
                c1 = ROOT.TCanvas("c1");
                l = ROOT.TLegend(0.2,0.68,0.9,0.91)
                l.SetNColumns(2)
                l.SetFillStyle(0)
                l.SetShadowColor(ROOT.kWhite)
                l.SetBorderSize(0)

                c1.SetLogy(logY)
                for i_histo, histo in enumerate(reversed(histos)):
                    histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
                    histo.GetYaxis().SetTitle("shape wrt. SM")
                    if i_histo == 0:
                        histo.Draw('hist')
                        histo.GetYaxis().SetRangeUser( (0.01 if logY else 0), (10 if logY else 2))
                        histo.Draw('hist')
                    else:
                        histo.Draw('histsame')
                    l.AddEntry(histo, histo.legendText)
                    c1.SetLogy(logY)
                l.Draw()

                plot_directory_ = os.path.join( plot_directory, "shape_plots", "log" if logY else "lin" )
                helpers.copyIndexPHP( plot_directory_ )
                c1.Print( os.path.join( plot_directory_, feature+postfix+'.png' ))

print ("Done with plots")
syncer.sync()

# GIF animation
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.06)
#nn_model = "ctGIm_lin" 
#epochs   = reversed(range(0, 160,10))
#derivatives = [('ctGIm', ), ('ctGIm', 'ctGIm')]

#nn_model = "ctGIm_lin_1" 
#epochs   = reversed(range(0, 251,50))
derivatives = [(args.WC, ) ]

#predictions =  ["ctGIm_lin_epoch_%i"%i for i in range(0,161,10)] #forgot to remove th lin from the name
predictions =  ["%s_epoch_%i"(args.model, %i) for i in args.epochs] #forgot to remove th lin from the name
data_generator = model.data( input_files = args.input_files, branches = model.features + model.observers + predictions )

# colors
color = {}
i_lin, i_diag, i_mixed = 0,0,0
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

for epoch in epochs:
    predictions = data_generator.vector_branch(nn_model+'_epoch_%i'%epoch)
    if predictions.ndim==1:
        predictions=predictions.reshape(-1,1) 

    w0 = weights[()]

    # 2D plots for convergence + animation
    th2d = {}
    th1d_pred = {}
    th1d_truth= {}
    for i_der, der in enumerate( derivatives ):
        truth_ratio = weights[der]/w0
        quantiles = np.quantile(np.concatenate( (truth_ratio, predictions[:,i_der] ) ), q=(0.01,1-0.01))

        if len(der)==2: #quadratic
            binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
        else:
            binning = np.linspace( quantiles[0], quantiles[1], 21 )

        th2d[der]      = helpers.make_TH2F( np.histogram2d( truth_ratio, predictions[:,i_der], bins = [binning, binning], weights=w0) )
        th1d_truth[der]= helpers.make_TH1F( np.histogram( truth_ratio, bins = binning, weights=w0) )
        th1d_pred[der] = helpers.make_TH1F( np.histogram( predictions[:,i_der], bins = binning, weights=w0) )
        tex_name = "%s"%(",".join( der ))
        th2d[der].GetXaxis().SetTitle( tex_name + " truth" )
        th2d[der].GetYaxis().SetTitle( tex_name + " prediction" )
        th1d_pred[der].GetXaxis().SetTitle( tex_name + " prediction" )
        th1d_pred[der].GetYaxis().SetTitle( "Number of Events" )
        th1d_truth[der].GetXaxis().SetTitle( tex_name + " truth" )
        th1d_truth[der].GetYaxis().SetTitle( "Number of Events" )

        th1d_truth[der].SetLineColor(ROOT.kBlack)
        th1d_truth[der].SetMarkerColor(ROOT.kBlack)
        th1d_truth[der].SetMarkerStyle(0)
        th1d_truth[der].SetLineWidth(2)
        th1d_truth[der].SetLineStyle(ROOT.kDashed)
        th1d_pred[der].SetLineColor(ROOT.kBlack)
        th1d_pred[der].SetMarkerColor(ROOT.kBlack)
        th1d_pred[der].SetMarkerStyle(0)
        th1d_pred[der].SetLineWidth(2)

    n_pads = len(derivatives)
    n_col  = len(derivatives) 
    n_rows = 2
    #for logZ in [False, True]:
    #    c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
    #    c1.Divide(n_col,n_rows)

    #    for i_der, der in enumerate(derivatives):

    #        c1.cd(i_der+1)
    #        ROOT.gStyle.SetOptStat(0)
    #        th2d[der].Draw("COLZ")
    #        ROOT.gPad.SetLogz(logZ)

    #    lines = [ (0.29, 0.9, 'N_{B} =%5i'%( epoch )) ]
    #    drawObjects = [ tex.DrawLatex(*line) for line in lines ]
    #    for o in drawObjects:
    #        o.Draw()

    #    for i_der, der in enumerate(derivatives):
    #        c1.cd(i_der+1+len(derivatives))
    #        l = ROOT.TLegend(0.6,0.75,0.9,0.9)
    #        stuff.append(l)
    #        l.SetNColumns(1)
    #        l.SetFillStyle(0)
    #        l.SetShadowColor(ROOT.kWhite)
    #        l.SetBorderSize(0)
    #        l.AddEntry( th1d_truth[der], "R("+tex_name+")")
    #        l.AddEntry( th1d_pred[der],  "#hat{R}("+tex_name+")")
    #        ROOT.gStyle.SetOptStat(0)
    #        th1d_pred[der].Draw("hist")
    #        th1d_truth[der].Draw("histsame")
    #        ROOT.gPad.SetLogy(logZ)
    #        l.Draw()


    #    plot_directory_ = os.path.join( plot_directory, "training_plots", nn_model, "log" if logZ else "lin" )
    #    if not os.path.isdir(plot_directory_):
    #        try:
    #            os.makedirs( plot_directory_ )
    #        except IOError:
    #            pass
    #    helpers.copyIndexPHP( plot_directory_ )
    #    c1.Print( os.path.join( plot_directory_, "training_2D_epoch_%05i.png"%(epoch) ) )
    #    syncer.makeRemoteGif(plot_directory_, pattern="training_2D_epoch_*.png", name="training_2D_epoch" )

    for observables, features, postfix in [
        ( model.observers if hasattr(model, "observers") else [], observers, "_observers"),
        ( model.features, features, ""),
            ]:
        if len(observables)==0: continue
        h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
        wp_pred = np.multiply(w0[:,np.newaxis], predictions)
        for i_feature, feature in enumerate(observables):
            # root style binning
            binning     = model.plot_options[feature]['binning']
            # linspace binning
            lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
            #digitize feature
            binned      = np.digitize(features[:,i_feature], lin_binning[feature] )
            # for each digit, create a mask to select the corresponding event in the bin (e.g. test_features[mask[0]] selects features in the first bin
            mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )
            h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
            h_derivative_prediction = np.array([ wp_pred[m].sum(axis=0) for m in mask])
            h_derivative_truth      = np.array([ (np.transpose(np.array([(weights[der] if der in weights else weights[tuple(reversed(der))]) for der in derivatives])))[m].sum(axis=0) for m in mask])
            h_ratio_prediction[feature] = h_derivative_prediction/(h_w0[feature].reshape(-1,1))
            h_ratio_truth[feature]      = h_derivative_truth/(h_w0[feature].reshape(-1,1))
        del wp_pred

        # 1D feature plot animation
        n_pads = len(observables)+1
        n_col  = int(sqrt(n_pads))
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

            for i_feature, feature in enumerate(observables):

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

                    tex_name = "%s"%(",".join( der ))

                    if i_feature==0:
                        l.AddEntry( th1d_ratio_truth[der], "R("+tex_name+")")
                        l.AddEntry( th1d_ratio_pred[der],  "#hat{R}("+tex_name+")")

                if i_feature==0:
                    l.AddEntry( th1d_yield, "yield (SM)")

                max_ = max( map( lambda h:h.GetMaximum(), list(th1d_ratio_truth.values())+list(th1d_ratio_pred.values()) ))
                max_ = 10**(1.5)*max_ if logY else 1.5*max_
                min_ = min( map( lambda h:h.GetMinimum(), list(th1d_ratio_truth.values())+list(th1d_ratio_pred.values()) ))
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

            c1.cd(len(observables)+1)
            l.Draw()

            lines = [ (0.29, 0.9, 'N_{B} =%5i'%( epoch )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            plot_directory_ = os.path.join( plot_directory, "training_plots", nn_model, "log" if logY else "lin" )
            if not os.path.isdir(plot_directory_):
                try:
                    os.makedirs( plot_directory_ )
                except IOError:
                    pass
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, epoch) ) )
            syncer.makeRemoteGif(plot_directory_, pattern="epoch%s_*.png"%postfix, name="epoch%s"%postfix )
        syncer.sync()
