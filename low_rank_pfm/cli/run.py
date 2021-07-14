# -*- coding: utf-8 -*-
"""Parser for phys2bids."""


import argparse

from low_rank_pfm import __version__


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Argument:")
    required.add_argument(
        "-i",
        "--input",
        dest="data_filename",
        type=str,
        nargs="+",
        help="The name of the file containing fMRI data. ",
        required=True,
    )
    required.add_argument(
        "-m",
        "--mask",
        dest="mask_filename",
        type=str,
        help="The name of the file containing the mask for " "the fMRI data. ",
        required=True,
    )
    required.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        type=str,
        help="The name of the output file with no extension.",
        required=True,
    )
    required.add_argument(
        "-tr",
        dest="tr",
        type=float,
        help="TR of the fMRI data acquisition.",
        required=True,
    )
    optional.add_argument(
        "-te",
        dest="te",
        nargs="*",
        type=float,
        help="List with TE of the fMRI data acquisition.",
        default=[0],
    )
    optional.add_argument(
        "-thr",
        dest="thr",
        type=float,
        help="Threshold to be used on debiasing step. Default = 0.01",
        default=1e-2,
    )
    optional.add_argument(
        "-eigthr",
        dest="eigthr",
        type=float,
        help="Threshold to be used on eigen value selection. Default = 0.25",
        default=0.25,
    )
    optional.add_argument(
        "-l",
        "--lambda",
        dest="lambda_weight",
        type=float,
        help="Weight to multiply noise estimation for regularization. " "Default = 1.1",
        default=1.1,
    )
    optional.add_argument(
        "-g",
        "--group",
        dest="group",
        type=float,
        help="Weight of the grouping in space (we suggest not going "
        "higher than 0.3). Default = 0.",
        default=0,
    )
    optional.add_argument(
        "-d",
        "--debias",
        dest="do_debias",
        action="store_true",
        help="Perform debiasing step. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-pfm",
        "--pfm",
        dest="is_pfm",
        action="store_true",
        help="Use original PFM formulation without low-rank. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-crit",
        "--criteria",
        dest="lambda_crit",
        type=str,
        choices=["mad", "mad_update", "ut", "lut", "pcg"],
        help="Criteria with which lambda is selected to estimate activity-inducing and innovation signals.",
        default="mad_update",
    )
    optional.add_argument(
        "-v", "--version", action="version", version=("%(prog)s " + __version__)
    )

    parser._action_groups.append(optional)

    return parser


if __name__ == "__main__":
    raise RuntimeError(
        "low_rank_pfm/cli/run.py should not be run directly;\n"
        "Please `pip install` low_rank_pfm and use the "
        "`low_rank_pfm` command"
    )
