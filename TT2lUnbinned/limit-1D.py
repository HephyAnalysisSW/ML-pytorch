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

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument("--plot_directory",     action="store",      default="limit-1D",     help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default="v2", type=str,  help="prefix")

#argParser.add_argument("--wc",                action="store",      default = "ctGRe", choices = ["cQj18", "cQj38", "ctGIm", "ctGRe", "ctj8"], help="Which wilson coefficient?")


args = argParser.parse_args()

#Logger
import tools.logger as logger_
logger = logger_.get_logger(args.logLevel, logFile = None )

                #( " t kin.", ROOT.kBlue, ["True", "False", "False", "False"]),
                #( "+l kin",  ROOT.kGreen, ["True", "True", "False", "False"]),
                #( "+ CA",    ROOT.kMagenta, ["True", "True", "True", "False"]),
                #( "+ SC",    ROOT.kRed, ["True", "True", "True", "True"]),
                #( "all",     ROOT.kBlack, ["False", "False", "False", "False"]),

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory)#, args.physics_model )
os.makedirs( plot_directory, exist_ok=True)

for wc in ["cQj18", "cQj38", "ctGIm", "ctGRe", "ctj8"]:
    c1 = ROOT.TCanvas()

    first = True
    stuff = []

    l = ROOT.TLegend(0.2,0.75,0.9,0.91)
    l.SetNColumns(2)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)

    for legendText, color, (TK, LK, CA, SC) in [ 
                ( "+ CA",    ROOT.kMagenta, ["False", "False", "True", "False"]),
                ( "+ SC",    ROOT.kRed,     ["False", "False", "True", "True"]),
                ( " t kin.", ROOT.kBlue,    ["True", "False", "True", "True"]),
                ( "+l kin",  ROOT.kGreen,   ["True", "True", "True", "True"]),
                ( "all",     ROOT.kBlack,   ["False", "False", "False", "False"]),
            ]:
                filename = os.path.join( user.results_directory, "TT2lUnbinned", "multiBit_TT2lUnbinned_TK_%s_LK_%s_CA_%s_SC_%s_v1_coeffs_ctGRe_ctGIm_cQj18_cQj38_ctj8_nTraining_-1_nTrees_300/%s.pkl"% (TK, LK, CA, SC, wc) )
                try: 
                    results = pickle.load( open( filename, 'rb'))
                except FileNotFoundError:
                    continue

                tgr = helpers.make_TGraph([(r['val'], r['postfit']) for r in results])

                tgr.SetLineWidth(2)
                tgr.SetLineColor(color)
                tgr.SetMarkerColor(color)
                tgr.SetMarkerStyle(0)
                if first:
                    first = False
                    tgr.Draw("AL")
                else:
                    tgr.Draw("L")
                tgr.GetYaxis().SetTitle( "-2 #Delta log L" )
                tgr.GetXaxis().SetTitle( wc )

                l.AddEntry( tgr, legendText )
                
                stuff.append( tgr )
    l.Draw()

    c1.Print(os.path.join(plot_directory, wc + '.png'))

helpers.copyIndexPHP( plot_directory )
syncer.sync()
