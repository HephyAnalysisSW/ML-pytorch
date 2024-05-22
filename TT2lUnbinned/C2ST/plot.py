import ROOT
import glob
import numpy as np
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import tools.syncer as syncer
import tools.user as user

def load( file ):
    try:
        return pickle.load(open(file, 'rb'))
    except EOFError:
        pass

null   = list(map( load, glob.glob("/groups/hephy/cms/robert.schoefbeck/NN/results/C2ST/TTLep_bTagSys_paper_v4_for_paper/removePred_[0-9]*.pkl") ))

meas   = load("/groups/hephy/cms/robert.schoefbeck/NN/results/C2ST/TTLep_bTagSys_paper_v4_for_paper/removePred_notShuffled.pkl")
before = load("/groups/hephy/cms/robert.schoefbeck/NN/results/C2ST/TTLep_bTagSys_paper_v4_for_paper/notShuffled.pkl")

Nbins=100
h = ROOT.TH1F("meas", "meas",  Nbins, 0.4985, 0.5045)
h1 = ROOT.TH1F("meas", "meas", Nbins, 0.4985, 0.5045)
h2 = ROOT.TH1F("meas", "meas", Nbins, 0.4985, 0.5045)

for n in null:
    h.Fill(n['accuracy'])

h.GetXaxis().SetTitle("accuracy")
h.SetLineWidth(2)
c1 = ROOT.TCanvas()

h.Draw()
h.GetXaxis().SetLabelSize(0.045)
c1.Update()
l=c1.GetUymax()

l1 = ROOT.TLine(meas['accuracy'], 0 ,meas['accuracy'],l)
h1.SetLineColor(ROOT.kWaterMelon)

l2 = ROOT.TLine(before['accuracy'], 0 ,before['accuracy'],l)
h2.SetLineColor(ROOT.kRed)

l1.SetLineWidth(2)
l2.SetLineWidth(2)
h1.SetLineWidth(2)
h2.SetLineWidth(2)

l1.SetLineColor(ROOT.kOrange+7)
l2.SetLineColor(ROOT.kOrange+7)
l2.SetLineStyle(ROOT.kDashed)
h1.SetLineColor(ROOT.kOrange+7)
h2.SetLineColor(ROOT.kOrange+7)
h2.SetLineStyle(ROOT.kDashed)

h .SetMarkerStyle(0)
h1 .SetMarkerStyle(0)
h1 .SetMarkerColor(ROOT.kOrange+7)
h2 .SetMarkerStyle(0)
h2 .SetMarkerColor(ROOT.kOrange+7)

l1.Draw("same")
l2.Draw("same")

legend = ROOT.TLegend(0.45,0.65,0.85,0.9)
legend.SetFillStyle(0)
legend.SetShadowColor(ROOT.kWhite)
legend.SetBorderSize(0)

legend.AddEntry( h, "randomized")
legend.AddEntry( h1, "reweighted")
legend.AddEntry( h2, "not reweighted")

legend.Draw()

c1.Print(os.path.join(user.plot_directory, "C2ST", "results", "TTLep_bTagSys_paper_v4_for_paper.png"))
c1.Print(os.path.join(user.plot_directory, "C2ST", "results", "TTLep_bTagSys_paper_v4_for_paper.pdf"))
#c1.Print(user.plot_directory, "C2ST", "results", "TTLep_bTagSys_paper_v4_for_paper.root")

syncer.sync()
