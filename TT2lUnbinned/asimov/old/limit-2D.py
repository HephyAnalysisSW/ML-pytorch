# Standard imports
import ROOT
import numpy as np
import math
import array
import sys, os, copy
import operator
import pickle

sys.path.insert(0, '..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers


#ROOT.gStyle.SetPalette(ROOT.kBird)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.gStyle.SetPalette(ROOT.kRainBow)

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument("--plot_directory",     action="store",      default="limit-2D",     help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default="v2", type=str,  help="prefix")

args = argParser.parse_args()

isPrefit = "prefit" in args.prefix.lower()

#Logger
import tools.logger as logger_
logger = logger_.get_logger(args.logLevel, logFile = None )

                #( " t kin.", ROOT.kBlue, ["True", "False", "False", "False"]),
                #( "+l kin",  ROOT.kGreen, ["True", "True", "False", "False"]),
                #( "+ CA",    ROOT.kMagenta, ["True", "True", "True", "False"]),
                #( "+ SC",    ROOT.kRed, ["True", "True", "True", "True"]),
                #( "all",     ROOT.kBlack, ["False", "False", "False", "False"]),

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix)#, args.physics_model )
os.makedirs( plot_directory, exist_ok=True)

import data_models.plot_options as plot_options

stuff = []

for wc1 in ["cQj18", "cQj38", "ctGIm", "ctGRe", "ctj8"]:
    for wc2 in ["cQj18", "cQj38", "ctGIm", "ctGRe", "ctj8"]:

        #l = ROOT.TLegend(0.2,0.75,0.9,0.91)
        #l.SetNColumns(2)
        #l.SetFillStyle(0)
        #l.SetShadowColor(ROOT.kWhite)
        #l.SetBorderSize(0)

        for TK in ["False", "True"]:
            for LK in ["False", "True"]:
                for CA in ["False", "True"]:
                    for SC in ["False", "True"]:

                        
                        filename = os.path.join( "multiBit_TT2lUnbinned_TK_%s_LK_%s_CA_%s_SC_%s_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300/%s_vs_%s.pkl"% (TK, LK, CA, SC, wc1, wc2) )
                        print (filename)
                        try: 
                            results = pickle.load( open( os.path.join( user.results_directory, "TT2lUnbinned", args.prefix, filename), 'rb'))
                        except FileNotFoundError:
                            continue

                        c1 = ROOT.TCanvas()

                        x_vals = sorted(list(set([r['val1'] for r in results])))
                        y_vals = sorted(list(set([r['val2'] for r in results])))

                        h = ROOT.TH2F('l', 'l', len(x_vals)-1, array.array('d', x_vals), len(y_vals)-1, array.array('d', y_vals))

                        key = 'prefit' if isPrefit else 'postfit'

                        for r in results:
                            if r[key]>0 and r[key]<10**5:
                                h.SetBinContent( h.FindBin(r['val1'], r['val2']), r[key] )

                        contours = [2.28, 5.99]# (68%, 95%) for 2D
                        histsForCont = h.Clone()
                        histsForCont.SetContour(len(contours),array.array('d', contours))            
                        histsForCont.Draw("contzlist")
                        c1.Update()
                        conts = ROOT.gROOT.GetListOfSpecials().FindObject("contours")
                        cont_p1 = conts.At(0).Clone()
                        cont_p2 = conts.At(1).Clone()
                        #c_contlist = ((ctypes.c_double)*(len(contours)))(*contours)

                        h.Draw("COLZ")
                        h.GetXaxis().SetTitle( plot_options.tex[wc1] )
                        h.GetYaxis().SetTitle( plot_options.tex[wc2] )

                        for conts in [cont_p2]:
                            for cont in conts:
                                cont.SetLineColor(ROOT.kOrange+7)
                                cont.SetLineWidth(3)
                    #            cont.SetLineStyle(7)
                                cont.Draw("same")
                        for conts in [cont_p1]:
                            for cont in conts:
                                cont.SetLineColor(ROOT.kSpring-1)
                                cont.SetLineWidth(3)
                    #            cont.SetLineStyle(7)
                                cont.Draw("same")

                        filename_ = os.path.join(plot_directory, filename.replace('.pkl', '.png'))

                        c1.SetRightMargin(0.15)

                        ROOT.gPad.Update()
                        palette = h.GetListOfFunctions().FindObject("palette")

                        palette.SetX1NDC(0.88)
                        palette.SetX2NDC(0.92)
                        palette.SetY1NDC(0.13)
                        palette.SetY2NDC(0.95)
                        ROOT.gPad.Modified()
                        ROOT.gPad.Update()

                        c1.Print( filename_ )

                        helpers.copyIndexPHP( os.path.dirname( filename_ ) )
    syncer.sync()
