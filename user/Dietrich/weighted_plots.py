#! /usr/bin/env python

import pathlib
import json
import logging
import yaml
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
DEFAULT_CONFIG = "weighted_plots.yaml"
DEFAULT_OUTPUT = "plots/weighted_plots.pdf"
BASE_DIR = pathlib.Path(__file__).parent.resolve()


@click.command
@click.option(
    "-c",
    "--config-file",
    default=DEFAULT_CONFIG,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option("-r", "--data-range", default=10)
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
    config_file: pathlib.Path, data_range: int, output: pathlib.Path, debug: bool
) -> None:

    if debug:
        log.setLevel(logging.DEBUG)

    log.info("Reading config from %s", config_file)
    with open(config_file, "r") as c:
        config = yaml.safe_load(c)

    eft_weights = EFTWeights(config["weights_info"])

    data_files = [config["data_path"].format(i) for i in range(data_range)]

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
    # weights2 = eft_weights(weight_coeff, ctBRe=1.0)
    # weights3 = eft_weights(weight_coeff, cHQ3=1.0)
    # weights4 = eft_weights(weight_coeff, cHt=1.0)
    # weights5 = eft_weights(weight_coeff, cHtbRe=1.0)
    # weights6 = eft_weights(weight_coeff, ctWIm=1.0)
    # weights7 = eft_weights(weight_coeff, ctBIm=1.0)
    # weights8 = eft_weights(weight_coeff, cHtbIm=1.0)
    log.debug("Sum of weights SM %f", np.sum(weights0))
    log.debug("Sum of weights ctWRe=1 %f", np.sum(weights1))

    with PdfPages(output) as pdf:
        for name in branches["gen_jet"]:
            mask = ~np.isnan(data[name])
            d = data[name][mask]
            w0 = weights0[mask]
            w1 = weights1[mask]
            # w2 = weights2[mask]
            # w3 = weights3[mask]
            # w4 = weights4[mask]
            # w5 = weights5[mask]
            # w6 = weights6[mask]
            # w7 = weights7[mask]
            # w8 = weights8[mask]
            fig, ax = plt.subplots(1, 2, layout="tight")
            h0, bins = np.histogram(d, bins=100, weights=w0)
            h1, bins = np.histogram(d, bins=bins, weights=w1)
            # h2, bins = np.histogram(d, bins=bins, weights=w2)
            # h3, bins = np.histogram(d, bins=bins, weights=w3)
            # h4, bins = np.histogram(d, bins=bins, weights=w4)
            # h5, bins = np.histogram(d, bins=bins, weights=w5)
            # h6, bins = np.histogram(d, bins=bins, weights=w6)
            # h7, bins = np.histogram(d, bins=bins, weights=w7)
            # h8, bins = np.histogram(d, bins=bins, weights=w8)
            ax[0].stairs(h0, bins, label="SM")
            ax[0].stairs(h1, bins, label="ctRWRe=1")
            # ax[0].stairs(h2, bins, label="ctBRe=1")
            # ax[0].stairs(h3, bins, label="cHQ3=1")
            # ax[0].stairs(h4, bins, label="cHt=1")
            # ax[0].stairs(h5, bins, label="cHtbRe=1")
            # ax[0].stairs(h6, bins, label="ctWIm=1")
            # ax[0].stairs(h7, bins, label="ctBIm=1")
            # ax[0].stairs(h8, bins, label="cHtbIm=1")
            ax[0].legend()
            ax[0].set_xlabel(name)

            d_name = name.replace("genJet_", "delphesJet_")
            mask = ~np.isnan(data[d_name])
            d = data[d_name][mask]
            w0 = weights0[mask]
            w1 = weights1[mask]
            # w2 = weights2[mask]
            # w3 = weights3[mask]
            # w4 = weights4[mask]
            # w5 = weights5[mask]
            # w6 = weights6[mask]
            # w7 = weights7[mask]
            # w8 = weights8[mask]
            h0, bins = np.histogram(d, bins=100, weights=w0)
            h1, bins = np.histogram(d, bins=bins, weights=w1)
            # h2, bins = np.histogram(d, bins=bins, weights=w2)
            # h3, bins = np.histogram(d, bins=bins, weights=w3)
            # h4, bins = np.histogram(d, bins=bins, weights=w4)
            # h5, bins = np.histogram(d, bins=bins, weights=w5)
            # h6, bins = np.histogram(d, bins=bins, weights=w6)
            # h7, bins = np.histogram(d, bins=bins, weights=w7)
            # h8, bins = np.histogram(d, bins=bins, weights=w8)
            ax[1].stairs(h0, bins, label="SM")
            ax[1].stairs(h1, bins, label="ctRWRe=1")
            # ax[1].stairs(h2, bins, label="ctBRe=1")
            # ax[1].stairs(h3, bins, label="cHQ3=1")
            # ax[1].stairs(h4, bins, label="cHt=1")
            # ax[1].stairs(h5, bins, label="cHtbRe=1")
            # ax[1].stairs(h6, bins, label="ctWIm=1")
            # ax[1].stairs(h7, bins, label="ctBIm=1")
            # ax[1].stairs(h8, bins, label="cHtbIm=1")
            ax[1].legend()
            ax[1].set_xlabel(d_name)

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
