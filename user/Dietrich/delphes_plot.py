#! /usr/bin/env python
"""Compare delphesJet output with genJets.
"""
import pathlib
import logging
import yaml
import os
from typing import Any

import click
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent.resolve()
SCRATCH_DIR = pathlib.Path("/scratch-cbe/users") / os.getlogin()

DEFAULT_CONFIG = BASE_DIR / "delphes_plot.yaml"
DEFAULT_OUTPUT = SCRATCH_DIR / "delphes_plot.pdf"


def fill1(
    d1: Any,
    d2: Any,
    weights: Any,
    nbins: int = 100,
    range: tuple[float, float] | None = None,
    angle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill two histograms with the same binning."""
    # normalise angles on [0, 2pi]
    if angle:
        d1 = np.where(d1 < 0, d1 + 2 * np.pi, d1)
        d2 = np.where(d2 < 0, d2 + 2 * np.pi, d2)
    if range is None:
        r1 = min(np.min(d1), np.min(d2))
        r2 = max(np.max(d1), np.max(d2))
    else:
        r1, r2 = range
    bins = np.linspace(r1, r2, nbins + 1)
    h1, _ = np.histogram(d1, bins, weights=weights)
    h2, _ = np.histogram(d2, bins=bins, weights=weights)

    return h1, h2, bins


def fill2(
    d1: Any,
    d2: Any,
    weights: Any,
    nbins: int = 100,
    drange: float | None = None,
    angle: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill histogram with diff of two arrays"""
    diff = d1 - d2
    # normalise angle diff to [-pi,pi]
    if angle:
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)

    # range assume that diff is around 0
    if drange is None:
        drange = max(-np.min(diff), np.max(diff))
    bins = np.linspace(-drange, drange, nbins + 1)
    h, _ = np.histogram(diff, bins, weights=weights)

    return h, bins


def fill3(
    d1: Any,
    d2: Any,
    weights: Any,
    bins: tuple[np.ndarray, np.ndarray],
    angle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill 2d histogram."""
    # normalise angles on [0, 2pi]
    if angle:
        d1 = np.where(d1 < 0, d1 + 2 * np.pi, d1)
        d2 = np.where(d2 < 0, d2 + 2 * np.pi, d2)
    return np.histogram2d(d1, d2, bins, weights=weights)


@click.command
@click.option(
    "-c",
    "--config-file",
    default=DEFAULT_CONFIG,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "-n",
    "--nr-files",
    default=10,
    type=click.IntRange(0, 100, clamp=True),
    help="Number of files to read.",
)
@click.option(
    "-o",
    "--output",
    default=DEFAULT_OUTPUT,
    type=click.Path(
        file_okay=True, dir_okay=False, path_type=pathlib.Path, writable=True
    ),
)
@click.option("--debug/--no-debug", default=False, help="Enable DEBUG output.")
def main(
    config_file: pathlib.Path, nr_files: int, output: pathlib.Path, debug: bool
) -> None:
    """Compare delphesJet with genJet."""
    if debug:
        log.setLevel(logging.DEBUG)

    log.info("Reading config from %s", config_file)
    with open(config_file, "r") as c:
        config = yaml.safe_load(c)

    log.info("Reading %d root files.", nr_files)
    data_files = [config["data_path"].format(i) for i in range(nr_files)]

    delphes_branches = [item["name"] for item in config["branches"]]
    genjet_branches = [b.replace("delphesJet", "genJet") for b in delphes_branches]
    all_branches = delphes_branches + genjet_branches + ["p_C"]

    cut = config.get("cut")
    if cut is None:
        log.info("All data is read.")
    else:
        log.info("Cut = %s", cut)
    data = uproot.concatenate(data_files, all_branches, cut)

    weights = data["p_C"][:, 0]

    with PdfPages(output) as pdf:
        for item in config["branches"]:
            dJet = item["name"]
            gJet = dJet.replace("delphesJet", "genJet")
            angle = dJet in ["delphesJet_phi"]
            # mask to remove nan's
            mask = ~np.isnan(data[dJet]) & ~np.isnan(data[gJet])
            h1, h2, bins = fill1(
                data[dJet][mask],
                data[gJet][mask],
                weights[mask],
                100,
                item.get("range"),
                angle,
            )
            h3, bins3 = fill2(
                data[dJet][mask],
                data[gJet][mask],
                weights[mask],
                100,
                item.get("drange"),
                angle,
            )

            h4, xbins, ybins = fill3(
                data[dJet][mask],
                data[gJet][mask],
                weights[mask],
                (bins, bins),
                angle
            )

            fig, ax = plt.subplot_mosaic(
                [["A", "A"], ["A", "A"], ["B", "C"]], figsize=(8, 8), layout="tight"
            )
            ax["A"].stairs(h1, bins, label="DelphesJet")
            ax["A"].stairs(h2, bins, label="GenJet")
            ax["A"].legend()

            ax["B"].stairs(h3, bins3)
            ax["B"].set_xlabel(f"{dJet} - {gJet}")

            ax["C"].pcolormesh(xbins, ybins, h4)
            ax["C"].set_ylabel("GenJet")
            ax["C"].set_xlabel("DelphesJet")

            fig.suptitle(dJet.replace("delphesJet_", ""))

            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    main()
