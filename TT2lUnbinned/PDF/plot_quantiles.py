#!/usr/bin/env python

# Standalone x vs mu_F coverage plot, using the same data_model/loading pattern as pdf_bit_training.py

import ROOT
import numpy as np
import os, sys, pickle
from math import log10
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# ROOT / style
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

# Helpers & infra
from tools import helpers
import tools.syncer as syncer
import tools.user as user

from data_models.plot_options import plot_options

# Parser
import argparse
argParser = argparse.ArgumentParser(description="x–muF coverage plot")
argParser.add_argument("--plot_directory",  action="store", default="PDF_xmuF_map", help="plot sub-directory")
argParser.add_argument("--model",           action="store", default="TT2l_PDF",     help="which data_model to load")
argParser.add_argument("--nTraining",       action="store", default=-1, type=int,   help="number of events to read (-1 = all)")
args, extra = argParser.parse_known_args(sys.argv[1:])

# Extra args passthrough to model.multi_bit_cfg (kept for compatibility with your workflow)
def parse_value(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

extra_args, key = {}, None
for arg in extra:
    if arg.startswith("--"):
        key = arg.lstrip("-"); extra_args[key] = True
    else:
        if isinstance(extra_args[key], list):
            extra_args[key].append(parse_value(arg))
        else:
            extra_args[key] = [parse_value(arg)]
for k, v in list(extra_args.items()):
    if isinstance(v, list) and len(v) == 1:
        extra_args[k] = v[0]

# Load model module exactly like in training
exec(f"import data_models.{args.model} as model")
model.multi_bit_cfg.update(extra_args)
data_model = model.DataModel(
    top_kinematics      = False,
    lepton_kinematics   = False,
    asymmetry           = False,
    spin_correlation    = False,
)

# Plot directory (same pattern as training)
plot_directory = os.path.join(user.plot_directory, args.plot_directory, args.model)
os.makedirs(plot_directory, exist_ok=True)

# Load feature/weight/observer arrays
training_features, training_weights, training_observers = data_model.getEvents(args.nTraining, return_observers=True)

# --- Extract observers & helpers (as in earlier scripts) ---
obs_names = getattr(model, "observers", ["Generator_x1","Generator_x2","Generator_id1","Generator_id2","Generator_scalePDF"])
ix_x1  = obs_names.index("Generator_x1")
ix_x2  = obs_names.index("Generator_x2")
ix_id1 = obs_names.index("Generator_id1")
ix_id2 = obs_names.index("Generator_id2")

x1   = training_observers[:, ix_x1]
x2   = training_observers[:, ix_x2]
id1  = training_observers[:, ix_id1].astype(int)
id2  = training_observers[:, ix_id2].astype(int)
w0   = training_weights[()]   # SM/base weights

# ---------------- Process-class masks ----------------
# gg fusion, qg mixed (either leg gluon), and q qbar annihilation
GG = (id1 == 21) & (id2 == 21)
QG = ((id1 == 21) & np.isin(np.abs(id2), [1,2,3,4,5,6])) | ((id2 == 21) & np.isin(np.abs(id1), [1,2,3,4,5,6]))
QQ = (np.isin(np.abs(id1), [1,2,3,4,5,6]) & np.isin(np.abs(id2), [1,2,3,4,5,6]) & (id1 * id2 < 0))  # q qbar

# For the mixed (qg) case: full-length per-leg x arrays (NaN where not applicable)
x_gluon_leg = np.where(id1 == 21, x1, np.where(id2 == 21, x2, np.nan))
x_quark_leg = np.where(id1 != 21, x1, np.where(id2 != 21, x2, np.nan))

# ---------------- Utilities ----------------
def make_th1_feature(mask, feature_idx, feature_name, color, title_suffix=""):
    """Weighted 1D feature histogram for a boolean mask."""
    bins = plot_options[feature_name]['binning']
    arr  = training_features[:, feature_idx][mask]
    ww   = w0[mask]
    h    = helpers.make_TH1F(np.histogram(arr, np.linspace(bins[1], bins[2], bins[0] + 1), weights=ww))
    h.SetDirectory(0)
    h.SetLineColor(color); h.SetLineWidth(2); h.SetMarkerStyle(0)
    if title_suffix:
        h.SetTitle(title_suffix)
    return h

def quantiles_per_bin(feature_idx, feature_name, values_full, weights_full,
                      base_mask=None, quantiles=(0.05, 0.32, 0.50, 0.68, 0.95)):
    """
    Compute weighted quantiles of 'values_full' in bins of the feature.
    All inputs are full-length (one per event). Use 'base_mask' to restrict.
    Returns (centers, edges, [TGraph for each quantile]).
    """
    bins    = plot_options[feature_name]['binning']
    edges   = np.linspace(bins[1], bins[2], bins[0] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fvals = training_features[:, feature_idx]
    if base_mask is None:
        base_mask = np.ones_like(weights_full, dtype=bool)

    valid = base_mask & np.isfinite(values_full) & np.isfinite(weights_full)

    graphs = []
    for q in quantiles:
        xs, ys = [], []
        for i in range(len(edges) - 1):
            sel_bin = valid & (fvals >= edges[i]) & (fvals < edges[i + 1])
            if not np.any(sel_bin):
                continue
            vy = values_full[sel_bin]
            wy = weights_full[sel_bin]
            if vy.size == 0 or wy.sum() <= 0:
                continue

            order     = np.argsort(vy)
            vy_sorted = vy[order]
            wy_sorted = wy[order]
            cdf       = np.cumsum(wy_sorted) / np.sum(wy_sorted)
            yq        = np.interp(q, cdf, vy_sorted)

            xs.append(centers[i]); ys.append(yq)

        graphs.append(ROOT.TGraph(len(xs), np.array(xs, dtype='d'), np.array(ys, dtype='d')))
    return centers, edges, graphs

# ---------------- Style / colors ----------------
col_gg = ROOT.kGreen + 2
col_qg = ROOT.kBlue  + 1
col_qq = ROOT.kRed   + 1
quant_cols = [ROOT.kGray+2, ROOT.kAzure+2, ROOT.kMagenta+1, ROOT.kOrange+1, ROOT.kCyan+2]

# ---------------- Output dir ----------------
outdir = os.path.join(plot_directory, "feature_panels_ttbar")
os.makedirs(outdir, exist_ok=True)
helpers.copyIndexPHP(outdir)

# ================== LOOP OVER FEATURES ==================
for feature_idx, feature_name in enumerate(data_model.feature_names):
    bins   = plot_options[feature_name]['binning']
    edges  = np.linspace(bins[1], bins[2], bins[0] + 1)

    c = ROOT.TCanvas(f"c_{feature_name}_panel", "", 2000, 520)
    c.Divide(4, 1)

    # -------- Pad 1: distribution split into gg, qg, qq --------
    c.cd(1)
    ROOT.gPad.SetLogy(True)

    h_gg = make_th1_feature(GG, feature_idx, feature_name, col_gg, "gg")
    h_qg = make_th1_feature(QG, feature_idx, feature_name, col_qg, "qg")
    h_qq = make_th1_feature(QQ, feature_idx, feature_name, col_qq, "q#bar{q}")

    h_gg.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    h_gg.GetYaxis().SetTitle("events (weighted)")

    maxy = max(h.GetMaximum() for h in (h_gg, h_qg, h_qq)) * 5.0
    h_gg.GetYaxis().SetRangeUser(max(1e-9, 0.5), maxy)

    h_gg.Draw("hist")
    h_qg.Draw("hist same")
    h_qq.Draw("hist same")

    # horizontal (3 columns) legend at the top
    leg1 = ROOT.TLegend(0.18, 0.86, 1, 0.96)
    leg1.SetNColumns(3)
    leg1.SetFillStyle(0)
    leg1.SetBorderSize(0)
    leg1.SetTextSize(0.035)

    leg1.AddEntry(h_gg, "gg", "l")
    leg1.AddEntry(h_qg, "qg", "l")
    leg1.AddEntry(h_qq, "q#bar{q}", "l")
    leg1.Draw()

    # -------- Pad 2: quantiles of x1 for gg vs feature --------
    c.cd(2)
    ROOT.gPad.SetGridy(True)

    _, edges2, graphs2 = quantiles_per_bin(feature_idx, feature_name, x1, w0, base_mask=GG)
    frame2 = ROOT.TH1F(f"frame2_{feature_name}", "", len(edges2) - 1, edges2)
    frame2.SetMinimum(1e-6); frame2.SetMaximum(1.0)
    frame2.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame2.GetYaxis().SetTitle("x_{1} quantiles (gg)")
    frame2.Draw()
    for i, gr in enumerate(graphs2):
        gr.SetLineColor(quant_cols[i % len(quant_cols)])
        gr.SetLineWidth(2)
        gr.Draw("L SAME")

    # -------- Pad 3: quantiles for qg (two sets: gluon-leg solid, quark-leg dashed) --------
    c.cd(3)
    ROOT.gPad.SetGridy(True)

    _, edges3, graphs_g = quantiles_per_bin(feature_idx, feature_name, x_gluon_leg, w0, base_mask=QG)
    _, _,      graphs_q = quantiles_per_bin(feature_idx, feature_name, x_quark_leg, w0, base_mask=QG)

    frame3 = ROOT.TH1F(f"frame3_{feature_name}", "", len(edges3) - 1, edges3)
    frame3.SetMinimum(1e-6); frame3.SetMaximum(1.0)
    frame3.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame3.GetYaxis().SetTitle("x quantiles (qg)")
    frame3.Draw()

    for i, gr in enumerate(graphs_g):
        gr.SetLineColor(quant_cols[i % len(quant_cols)])
        gr.SetLineWidth(2)
        gr.SetLineStyle(1)
        gr.Draw("L SAME")
    for i, gr in enumerate(graphs_q):
        gr.SetLineColor(quant_cols[i % len(quant_cols)])
        gr.SetLineWidth(2)
        gr.SetLineStyle(7)
        gr.Draw("L SAME")

    # --- Pad 3 legend (use dummy objects, not None) ---
    leg3 = ROOT.TLegend(0.25, 0.82, 1., 0.96)
    leg3.SetNColumns(2); leg3.SetFillStyle(0); leg3.SetBorderSize(0)

    # Dummy line objects to illustrate styles
    dummy_solid  = ROOT.TH1F(f"dummy_solid_{feature_name}",  "", 1, 0, 1)
    dummy_dashed = ROOT.TH1F(f"dummy_dashed_{feature_name}", "", 1, 0, 1)
    dummy_solid.SetLineColor(ROOT.kBlack);  dummy_solid.SetLineStyle(1); dummy_solid.SetLineWidth(2)
    dummy_dashed.SetLineColor(ROOT.kBlack); dummy_dashed.SetLineStyle(7); dummy_dashed.SetLineWidth(2)

    # Keep references so they don't get garbage-collected before Draw()
    # (use an existing list like `stuff` if you have one, else make one)
    try:
        stuff.append(dummy_solid); stuff.append(dummy_dashed)
    except NameError:
        stuff = [dummy_solid, dummy_dashed]

    leg3.AddEntry(dummy_solid,  "solid: gluon-leg",  "l")
    leg3.AddEntry(dummy_dashed, "dashed: quark-leg", "l")
    leg3.Draw()


    # -------- Pad 4: quantiles of x1 for qq̄ vs feature --------
    c.cd(4)
    ROOT.gPad.SetGridy(True)

    _, edges4, graphs4 = quantiles_per_bin(feature_idx, feature_name, x1, w0, base_mask=QQ)
    frame4 = ROOT.TH1F(f"frame4_{feature_name}", "", len(edges4) - 1, edges4)
    frame4.SetMinimum(1e-6); frame4.SetMaximum(1.0)
    frame4.GetXaxis().SetTitle(plot_options[feature_name]['tex'])
    frame4.GetYaxis().SetTitle("x_{1} quantiles (q#bar{q})")
    frame4.Draw()
    for i, gr in enumerate(graphs4):
        gr.SetLineColor(quant_cols[i % len(quant_cols)])
        gr.SetLineWidth(2)
        gr.Draw("L SAME")

    # --- Add TLatex labels to pads 2, 3, 4 ---
    for pad_idx, label in [(2, "gg"), (3, "gq"), (4, "q#bar{q}")]:
        c.cd(pad_idx)
        tex = ROOT.TLatex()
        tex.SetNDC()
        tex.SetTextFont(42)
        tex.SetTextSize(0.05)
        tex.DrawLatex(0.20, 0.88, label)
        # keep a Python ref so it doesn't get GC'd before saving
        try:
            stuff.append(tex)
        except NameError:
            pass

    # -------- Save canvas --------
    c.Print(os.path.join(outdir, f"{feature_name}_panel.pdf"))
    c.Print(os.path.join(outdir, f"{feature_name}_panel.png"))
    c.Close()

