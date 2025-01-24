"""
Creates "combine datacards" using hist.Hist templates, and
sets up data-driven QCD background estimate ('rhalphabet' method)

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
    https://github.com/farakiko/boostedhiggs/blob/main/combine/create_datacard.py

Author: Yuzhe Zhao
"""

from __future__ import division, print_function

import argparse
import json
import logging
import math
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import os, sys
# sys.path.append("/ospool/cms-user/yuzhe/BoostedHWW/rhaphabetlib/CMSSW_11_3_4/src/rhalphalib")
import rhalphalib as rl
# import hist
from hist import Hist
from pathlib import Path

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field

from collections import OrderedDict
from utils import blindBins, get_template, labels, samples, shape_to_num, sigs 

from datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    sum_templates,
    get_effect_updown,
    get_year_updown,
    rem_neg,
    qcd_scale_acc_dict,
    pdf_scale_acc_dict
)

rl.ParametericSample.PreferRooParametricHist = False
logging.basicConfig(level=logging.INFO)
import argparse

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


CMS_PARAMS_LABEL = "boosted_Wcb"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")
parser.add_argument("--model-name", default="HWWfhModel", type=str, help="output model name")
parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument("--epsilon", default=1e-2, type=float, help="epsilon to avoid numerical errs")
parser.add_argument(
    "--scale-templates", default=None, type=float, help="scale all templates for bias tests"
)
parser.add_argument(
    "--min-qcd-val", default=1e-4, type=float, help="clip the pass QCD to above a minimum value"
)
parser.add_argument(
    "--year",
    help="year",
    type=str,
    default="all",
    choices=["2016APV", "2016", "2017", "2018", "all"],
)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)

args = parser.parse_args()


mc_samples = OrderedDict(
    [
        ("b_top_matched_bqq", "b_top_matched_bqq"),
        ("b_top_matched_bq", "b_top_matched_bq"),
        ("b_w_matched_others", "b_w_matched_others"),
        ("b_w_matched_cb", "restb_w_matched_cb_bkg"),
        ("b_others","b_others"),
        ("WJets","WJets")
    ]
)
bg_keys = list(mc_samples.keys())
sig_keys = [
    "s_top_matched_bqq",
    "s_top_matched_bq",
    "s_w_matched_others",
    "s_w_matched_cb",
    "s_others",
]

sig_keys_dict = {
    "s_top_matched_bqq" : "s_top_matched_bqq",
    "s_top_matched_bq" : "s_top_matched_bq",
    "s_w_matched_others"  : "s_w_matched_others",
    "s_w_matched_cb"  : "s_w_matched_cb",
    "s_others" : "s_others",
}

for key in sig_keys:
    mc_samples[key] = sig_keys_dict[key]
    # temporary solution for WH/ZH/ttH as signal
    # mc_samples[key] = key + "_sig" if key in ["WH","ZH","ttH"] else key
        
all_mc = list(mc_samples.keys())
logging.info("all MC = %s" % all_mc)

# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    # none for now
}

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    
}

uncorr_year_shape_systs = {

}

shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if not syst.samples_corr:
        # separate nuisance param for each affected sample
        for sample in syst.samples:
            if sample not in mc_samples:
                continue #means MC name error
            shape_systs_dict[f"{skey}_{sample}"] = rl.NuisanceParameter(
                f"{syst.name}_{mc_samples[sample]}", "shape"
            )
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "shape")

def get_templates(
    templates_dir: Path,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""

    print("template_dir =",templates_dir )
    print("template =",templates_dir / f"templates.pkl" )
    with (templates_dir / f"templates.pkl").open("rb") as f:
        templates_dict = rem_neg(pkl.load(f))

    return templates_dict


def process_lp_systematics(lp_systematics: dict):
    """Get total uncertainties from per-year systs in ``systematics``"""
    for region in ["a","b"]:
        # already for all years
        nuisance_params[f"{CMS_PARAMS_LABEL}_lp_sf_region_{region}"].value = (
            1 + lp_systematics[region]
        )


def fill_regions(
    model: rl.Model,
    regions: List[str],
    templates_dict: Dict,
    templates_summed: Dict,
    mc_samples: Dict[str, str],
    nuisance_params: Dict[str, Syst],
    nuisance_params_dict: Dict[str, rl.NuisanceParameter],
    corr_year_shape_systs: Dict[str, Syst],
    uncorr_year_shape_systs: Dict[str, Syst],
    shape_systs_dict: Dict[str, rl.NuisanceParameter],
    bblite: bool = True,
):
    """Fill samples per region including given rate, shape and mcstats systematics.
    Ties "blinded" and "nonblinded" mc stats parameters together.

    Args:
        model (rl.Model): rhalphalib model
        regions (List[str]): list of regions to fill
        templates_dict (Dict): dictionary of all templates
        templates_summed (Dict): dictionary of templates summed across years
        templates: the combination the above two options
        mc_samples (Dict[str, str]): dict of mc samples and their names in the given templates -> card names
        nuisance_params (Dict[str, Tuple]): dict of nuisance parameter names and tuple of their
          (modifier, samples affected by it, value)
        nuisance_params_dict (Dict[str, rl.NuisanceParameter]): dict of nuisance parameter names
          and NuisanceParameter object
        corr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (correlated across years)
          and tuple of their (name in cards, samples affected by it)
        uncorr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (unccorrelated across years)
          and tuple of their (name in cards, samples affected by it)
        shape_systs_dict (Dict[str, rl.NuisanceParameter]): dict of shape syst names and
          NuisanceParameter object
        pass_only (List[str]): list of systematics which are only applied in the pass region(s)
        bblite (bool): use Barlow-Beeston-lite method or not (single mcstats param across MC samples)
        mX_bin (int): if doing 2D fit (for resonant), which mX bin to be filled
    """

    for region in regions:
        region_templates = templates_summed[region]

        # pass region = SR1a, SR1b, SR2a, SR2b, SR3a, SR3b, and same with "Blinded" suffix
        pass_region = False
        pass_regs = ["SR"]
        for pass_regi in pass_regs:
            if pass_regi in region: pass_region = True
        logging.info("starting region: %s" % region)
        ch = rl.Channel(region.replace("_", "")) 
        print(region.replace("_", ""))
        model.addChannel(ch)

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            if sample_name in sig_keys:
                if not pass_region:
                    #don't need to enter CR signal anyway
                    logging.info(f"\nSkipping {sample_name} in {region} region\n")
                    continue

            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name,:]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0) #select value >= 0

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
            logging.debug("error     : {errors}".format(errors=errors_nominal))

            #not used
            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info("setting autoMCStats for %s in %s" % (sample_name, region))
                # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
                region_name = region
                stats_sample_name = f"{CMS_PARAMS_LABEL}_{region_name}_{card_name}"
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this function uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            ch.addSample(sample)

        # we always use bblite method
        if bblite and args.mcstats:
            # pass
            # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
            channel_name = region 
            ch.autoMCStats(
                channel_name=f"{CMS_PARAMS_LABEL}_{channel_name}",
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed, not used
        ch.setObservation(region_templates["data", :])

def main(args):
    #print years

    # all SRs and CRs
    regions : List[str] = ["SR"]
    cur_dir = os.getcwd()
    print("current dir = ",cur_dir)
    
    #normalized the path
    args.templates_dir = Path(args.templates_dir)
    args.cards_dir = Path(args.cards_dir)
    
    # templates per region per year, templates per region summed across years
    templates_summed = get_templates(
        args.templates_dir
    )
        
    #apply lund plane sf
    model = rl.Model("HWWfullhad")
    #random template from which to extract shape vars
    sample_templates: Hist = templates_summed[next(iter(templates_summed.keys()))]
    
    #MH_Reco for full-hadronic boosted HWW
    shape_vars = [
        ShapeVar(name=axis.name, bins=axis.edges)
        for i, axis in enumerate(sample_templates.axes[1:]) #should be [1:] for boosted HWW analysis, because the 1st axes is mass
    ]
    # logging.info("shape_var = ",shape_vars)
    print("shape_var[0] info",shape_vars[0].name,shape_vars[0].scaled,shape_vars[0].bins)
    fill_args = [
        model,
        regions,
        None ,
        templates_summed,
        mc_samples,
        nuisance_params,
        nuisance_params_dict,
        corr_year_shape_systs,
        uncorr_year_shape_systs,
        shape_systs_dict,
        args.bblite,
    ]
    fit_args = [model, shape_vars,templates_summed, args.scale_templates, args.min_qcd_val]
    fill_regions(*fill_args)
    
    logging.info("rendering combine model")

    os.system(f"mkdir -p {args.cards_dir}")

    out_dir = (
        os.path.join(str(args.cards_dir), args.model_name)
        if args.model_name is not None
        else args.cards_dir
    )
    model.renderCombine(out_dir)

    with open(f"{out_dir}/model.pkl", "wb") as fout:
        pkl.dump(model, fout, 2)  # use python 2 compatible protocol
        
main(args)