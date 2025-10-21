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

# --- Extract observers ---
# observers = ["Generator_x1","Generator_x2","Generator_id1","Generator_id2","Generator_scalePDF"]
obs_names = getattr(model, "observers", ["Generator_x1","Generator_x2","Generator_id1","Generator_id2","Generator_scalePDF"])
idx_x1   = obs_names.index("Generator_x1")
idx_x2   = obs_names.index("Generator_x2")
idx_muF  = obs_names.index("Generator_scalePDF")
idx_id1  = obs_names.index("Generator_id1")
idx_id2  = obs_names.index("Generator_id2")

x1   = training_observers[:, idx_x1]
x2   = training_observers[:, idx_x2]
mu_F = training_observers[:, idx_muF]  # in GeV (generator factorization scale)
id1  = training_observers[:, idx_id1].astype(int)
id2  = training_observers[:, idx_id2].astype(int)

# Event weights (SM)
if () not in training_weights:
    raise RuntimeError("training_weights[()] not found. Expected SM/base weight under key ()")
w = training_weights[()]

# --- Define log-binning (robust to tails) ---
def robust_edges(arr, lo_pow_default, hi_pow_default, n_bins, is_x=False):
    # Use percentiles to avoid outliers; fall back to defaults if needed
    if arr.size == 0:
        if is_x:
            lo, hi = 10**(-6), 1.0
        else:
            lo, hi = 10**(lo_pow_default), 10**(hi_pow_default)
    else:
        qlo = np.percentile(arr,  0.5)
        qhi = np.percentile(arr, 99.5)
        if is_x:
            # clamp within (1e-6, 1)
            lo = max(1e-6, min(qlo, 0.5))
            hi = min(1.0,  max(qhi, 0.999999))
        else:
            # clamp within (1, 10^4) GeV
            lo = max(1.0,  min(qlo, 1e3))
            hi = min(1e4,  max(qhi, 2.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            # Fallback
            if is_x:
                lo, hi = 1e-6, 1.0
            else:
                lo, hi = 10**(lo_pow_default), 10**(hi_pow_default)
    return np.logspace(log10(lo), log10(hi), n_bins + 1)

# ---------- Overall 2D plot ----------
# Build doubled arrays: two entries per event -> (x1, muF) and (x2, muF)
x_all   = np.concatenate([x1, x2])
mu_all  = np.concatenate([mu_F, mu_F])
w_all   = np.concatenate([w,  w])

# Basic cleaning: positive x and mu_F
mask = (x_all > 0) & (x_all <= 1) & (mu_all > 0)
x_all, mu_all, w_all = x_all[mask], mu_all[mask], w_all[mask]

xbins = robust_edges(x_all,  -6, 0,  60, is_x=True)
mbins = robust_edges(mu_all,  0, 4,  80, is_x=False)

# Expand log axes by a factor 10 on both ends (x clamped to ≤ 1)
x_lo = max(1e-12, xbins[0] / 10.0)
x_hi = min(1.0,     xbins[-1] * 10.0)
xbins = np.logspace(log10(x_lo), log10(x_hi), len(xbins))

m_lo = max(1e-6,  mbins[0] / 10.0)
m_hi =            mbins[-1] * 10.0
mbins = np.logspace(log10(m_lo), log10(m_hi), len(mins)) if False else np.logspace(log10(m_lo), log10(m_hi), len(mbins))  # keep len(mbins)

# --- Fill TH2D with variable binning ---
h2 = ROOT.TH2D("h_x_muF", ";x (momentum fraction);#mu_{F}  [GeV]", len(xbins)-1, xbins, len(mbins)-1, mbins)
h2.Sumw2()

# Fast vectorized filling via numpy histogram2d, then transfer to TH2
H, _, _ = np.histogram2d(x_all, mu_all, bins=[xbins, mbins], weights=w_all)
for ix in range(1, len(xbins)):
    for iy in range(1, len(mbins)):
        h2.SetBinContent(ix, iy, H[ix-1, iy-1])

# --- Draw ---
c = ROOT.TCanvas("c_x_muF", "x vs muF", 800, 700)
c.SetRightMargin(0.15)
c.SetLogx(True)
c.SetLogy(True)
c.SetLogz(True)

h2.SetTitle("")
h2.GetZaxis().SetTitle("weighted event density")
h2.GetZaxis().SetTitleOffset(1.2)
h2.Draw("COLZ")

# Save
outdir = os.path.join(plot_directory, "x_muF_map")
os.makedirs(outdir, exist_ok=True)
helpers.copyIndexPHP(outdir)

png_name = os.path.join(outdir, "x_muF_density.png")
pdf_name = os.path.join(outdir, "x_muF_density.pdf")
c.Print(png_name)
c.Print(pdf_name)
c.Close()

print("Wrote:", png_name)
print("Wrote:", pdf_name)

# ---------- Per-PDG 2D plots (one plot per unique incoming parton id) ----------
outdir_pdg = os.path.join(plot_directory, "x_muF_map_byPDG")
os.makedirs(outdir_pdg, exist_ok=True)
helpers.copyIndexPHP(outdir_pdg)

unique_pdg = np.unique(np.concatenate([id1, id2]))

for pid in unique_pdg:
    # Select entries where this pid appears in either beam,
    # and add ONE entry per occurrence: (x1,muF) if id1==pid, (x2,muF) if id2==pid
    sel1 = (id1 == pid)
    sel2 = (id2 == pid)

    x_sel   = np.concatenate([x1[sel1], x2[sel2]])
    mu_sel  = np.concatenate([mu_F[sel1], mu_F[sel2]])
    w_sel   = np.concatenate([w[sel1],    w[sel2]])

    # Clean
    mask = (x_sel > 0) & (x_sel <= 1) & (mu_sel > 0)
    x_sel, mu_sel, w_sel = x_sel[mask], mu_sel[mask], w_sel[mask]
    if x_sel.size == 0:
        continue

    # Binning (robust) per PDG id
    xbins = robust_edges(x_sel, -6, 0, 60, is_x=True)
    mbins = robust_edges(mu_sel, 0, 4, 80, is_x=False)

    # Expand log axes by ×10 on both ends (x clamped to ≤ 1)
    x_lo = max(1e-12, xbins[0] / 10.0)
    x_hi = min(1.0,     xbins[-1] * 10.0)
    xbins = np.logspace(log10(x_lo), log10(x_hi), len(xbins))

    m_lo = max(1e-6,  mbins[0] / 10.0)
    m_hi =            mbins[-1] * 10.0
    mbins = np.logspace(log10(m_lo), log10(m_hi), len(mbins))

    # Histogram
    hname = f"h_x_muF_id_{pid}"
    h2pdg = ROOT.TH2D(hname, ";x (momentum fraction);#mu_{F}  [GeV]", len(xbins)-1, xbins, len(mbins)-1, mbins)
    h2pdg.Sumw2()

    H, _, _ = np.histogram2d(x_sel, mu_sel, bins=[xbins, mbins], weights=w_sel)
    for ix in range(1, len(xbins)):
        for iy in range(1, len(mbins)):
            h2pdg.SetBinContent(ix, iy, H[ix-1, iy-1])

    # Draw
    cpdg = ROOT.TCanvas(f"c_x_muF_id_{pid}", "x vs muF", 800, 700)
    cpdg.SetRightMargin(0.15)
    cpdg.SetLogx(True); cpdg.SetLogy(True); cpdg.SetLogz(True)

    h2pdg.SetTitle("")
    h2pdg.GetZaxis().SetTitle("weighted event density")
    h2pdg.GetZaxis().SetTitleOffset(1.2)
    h2pdg.Draw("COLZ")

    # Annotate with the PDG id
    tex = ROOT.TLatex()
    tex.SetNDC(); tex.SetTextSize(0.04); tex.SetTextAlign(11)
    tex.DrawLatex(0.15, 0.90, f"PDF id = {pid}")

    base = f"x_muF_density_id_{pid}"
    cpdg.Print(os.path.join(outdir_pdg, base + ".png"))
    cpdg.Print(os.path.join(outdir_pdg, base + ".pdf"))
    cpdg.Close()

    print("Wrote:", os.path.join(outdir_pdg, base + ".png"))
    print("Wrote:", os.path.join(outdir_pdg, base + ".pdf"))

# ---------- NEW: 1D x overlay by PDG id ----------
# Build global x-bins from all x entries (using doubled array from above)
xbins_1d = robust_edges(x_all, -6, 0, 80, is_x=True)
# Expand by ×10 on both ends, clamp to [1e-12, 1]
x1_lo = max(1e-12, xbins_1d[0] / 10.0)
x1_hi = min(1.0,     xbins_1d[-1] * 10.0)
xbins_1d = np.logspace(log10(x1_lo), log10(x1_hi), len(xbins_1d))

# Colors to cycle through
color_wheel = [
    ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kMagenta+1,
    ROOT.kOrange+1, ROOT.kCyan+2, ROOT.kViolet+2, ROOT.kAzure+2,
    ROOT.kPink+7, ROOT.kTeal+2, ROOT.kSpring+5, ROOT.kGray+2,
]

# Create canvas & legend
c1d = ROOT.TCanvas("c_x_overlay_pdg", "x overlay by PDG", 900, 700)
c1d.SetLogx(True)   # helpful for x spanning decades
c1d.SetLogy(True)   # requested: log-y

leg = ROOT.TLegend(0.12, 0.86, 0.95, 0.98)  # top band
leg.SetNColumns(4)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextSize(0.03)

hists = []
ymax = 0.0

for i, pid in enumerate(unique_pdg):
    sel1 = (id1 == pid)
    sel2 = (id2 == pid)
    x_sel = np.concatenate([x1[sel1], x2[sel2]])
    w_sel = np.concatenate([w[sel1],  w[sel2]])

    mask = (x_sel > 0) & (x_sel <= 1)
    x_sel, w_sel = x_sel[mask], w_sel[mask]
    if x_sel.size == 0:
        continue

    h = ROOT.TH1D(f"h_x_id_{pid}", ";x (momentum fraction);weighted events", len(xbins_1d)-1, xbins_1d)
    h.Sumw2()
    # Fill via numpy then transfer (faster & precise for variable bins)
    Hx, _ = np.histogram(x_sel, bins=xbins_1d, weights=w_sel)
    for ib in range(1, len(xbins_1d)):
        h.SetBinContent(ib, Hx[ib-1])

    col = color_wheel[i % len(color_wheel)]
    h.SetLineColor(col)
    h.SetMarkerColor(col)
    h.SetMarkerStyle(0)
    h.SetLineWidth(2)

    hists.append((pid, h))
    ymax = max(ymax, h.GetMaximum())

# Draw in one go
if hists:
    # set a sensible y-range for log scale
    ymin = max(1e-6, 0.5 * min([h.GetMinimum(0) or 1e-6 for _, h in hists]))
    ymax = ymax * 5.0

    for idx, (pid, h) in enumerate(hists):
        h.GetYaxis().SetRangeUser(ymin, ymax)
        h.Draw("HIST" if idx == 0 else "HIST SAME")
        leg.AddEntry(h, f"id = {pid}", "l")

    leg.Draw()

    out_overlay_png = os.path.join(outdir_pdg, "x_overlay_byPDG.png")
    out_overlay_pdf = os.path.join(outdir_pdg, "x_overlay_byPDG.pdf")
    c1d.Print(out_overlay_png)
    c1d.Print(out_overlay_pdf)
    print("Wrote:", out_overlay_png)
    print("Wrote:", out_overlay_pdf)

c1d.Close()

# Sync (as in your workflow)
syncer.sync()

