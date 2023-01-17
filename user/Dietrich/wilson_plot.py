#! /usr/bin/env python
"""Effect of wilson copefficients."""
import pathlib
import json
import logging
import yaml
import os
import sys

import itertools

import awkward as ak
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

DEFAULT_CONFIG = BASE_DIR / "wilson_plot.yaml"
DEFAULT_OUTPUT = SCRATCH_DIR / "wilson_plot.pdf"


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
    """Effect of wilson coefficients."""
    if debug:
        log.setLevel(logging.DEBUG)

    log.info("Reading config from %s", config_file)
    with open(config_file, "r") as c:
        config = yaml.safe_load(c)

    eft_weights = EFTWeights(BASE_DIR / config["weights_info"])

    data_files = [config["data_path"].format(i) for i in range(nr_files)]

    branches = config["branches"]
    all_branches = list(itertools.chain.from_iterable(branches.values()))

    cut = config.get("cut")
    if cut is None:
        log.info("All data is read.")
    else:
        log.info("Cut = %s", cut)
    data = uproot.concatenate(data_files, all_branches, cut)

    weight_coeff = data[config["branches"]["weight_coeff"][0]]

    weights0 = eft_weights(weight_coeff)
    weights1 = eft_weights(weight_coeff, ctWRe=1.0)
    log.debug("Sum of weights SM %f", np.sum(weights0))
    log.debug("Sum of weights ctWRe=1 %f", np.sum(weights1))

    with PdfPages(output) as pdf:
        for name in branches["gen_jet"]:
            mask = ~np.isnan(data[name])
            d = data[name][mask]
            w0 = weights0[mask]
            w1 = weights1[mask]
            fig, ax = plt.subplot_mosaic(
                [["A1", "B1"], ["A1", "B1"], ["A2", "B2"]], layout="tight"
            )
            h0, bins = np.histogram(d, bins=100, weights=w0)
            h1, bins = np.histogram(d, bins=bins, weights=w1)
            ax["A1"].stairs(h0, bins, label="SM")
            ax["A1"].stairs(h1, bins, label="ctRWRe=1")
            ax["A1"].legend()
            ax["A1"].set_xlabel(name)
            ax["A1"].set_yscale("log")

            hr = h1 / h0
            ax["A2"].plot((bins[:-1] + bins[1:]) / 2.0, hr, "bo", markersize=1)
            ax["A2"].set_ylim((0.5, 1.5))

            d_name = name.replace("genJet_", "delphesJet_")
            mask = ~np.isnan(data[d_name])
            d = data[d_name][mask]
            w0 = weights0[mask]
            w1 = weights1[mask]
            h0, bins = np.histogram(d, bins=100, weights=w0)
            h1, bins = np.histogram(d, bins=bins, weights=w1)
            ax["B1"].stairs(h0, bins, label="SM")
            ax["B1"].stairs(h1, bins, label="ctRWRe=1")
            ax["B1"].legend()
            ax["B1"].set_xlabel(d_name)
            ax["B1"].set_yscale("log")

            hr = h1 / h0
            ax["B2"].plot((bins[:-1] + bins[1:]) / 2.0, hr, "bo", markersize=1)
            ax["B2"].set_ylim((0.5, 1.5))

            pdf.savefig()
            plt.close()


class EFTWeights:

    order: int
    vars: list[str]
    id: list[str]
    ref_point: dict[str, float]

    def __init__(self, input: pathlib.Path) -> None:

        log.info("Reading weight info from %s", input)
        with open(input, "r") as inp:
            json_data = json.load(inp)

        try:
            data = json_data["rw_dict"]
        except KeyError:
            raise RuntimeError("No rw_dict found.")

        self.id = sorted(data.keys(), key=lambda x: data[x])

        # variables
        self.vars = list(data.keys())[0].split("_")[::2]
        log.debug("Variables: %s", ", ".join(self.vars))

        # for i,name in enumerate(self.id):
        #     log.debug("Weight %2.2d: %s", i, " - ".join([n[0] for n in name.split("_")[1::2]]))

        log.info("Found %d variables and %d weights", len(self.vars), len(self.id))

        # order
        try:
            self.order = json_data["order"]["order"]
        except KeyError:
            raise RuntimeError("No order found")

        # reference point
        rp: dict[str, float] = json_data.get("ref_point", {})
        self.ref_point = {v: rp.get(v, 0) for v in self.vars}

        # assert combinations
        # i = 0
        # for o in range(self.order + 1):
        #     for c in itertools.combinations_with_replacement(self.vars, o):
        #         print(c, self.id[i])
        #         i += 1

    def __call__(self, coeff: ak.highlevel.Array, **kwargs):

        for name in kwargs.keys():
            if name not in self.vars:
                log.error("Variable %s is not a valid eft parameter.")

        i = 0
        # order 0
        w = coeff[:, i]
        i += 1

        # order 1
        for v in self.vars:
            if v in kwargs:
                w = w + coeff[:, i] * kwargs[v]
            i += 1

        # order 2
        for v1, v2 in itertools.combinations_with_replacement(self.vars, 2):
            if v1 in kwargs and v2 in kwargs:
                w = w + coeff[:, i] * kwargs[v1] * kwargs[v2]
            i += 1

        return w


if __name__ == "__main__":
    main()
