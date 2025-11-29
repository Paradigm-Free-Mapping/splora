"""Parser for splora."""

import argparse

from splora import __version__


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
        help="The name of the file containing the mask for the fMRI data. ",
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
        "-d",
        "--dir",
        dest="out_dir",
        type=str,
        help="Output directory. Default is current.",
        default=".",
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
        "-block",
        "--block",
        dest="block_model",
        action="store_true",
        help="Estimate innovation signals. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-eigthr",
        dest="eigthr",
        type=float,
        help="Threshold to be used on eigen value selection. Default = 0.1",
        default=0.1,
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
        "--debias",
        dest="do_debias",
        action="store_true",
        help="Perform debiasing step. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-pfm",
        "--pfm",
        dest="pfm_only",
        action="store_true",
        help="Use original PFM formulation without low-rank. Default = False.",
        default=False,
    )
    optional.add_argument(
        "-crit",
        "--criteria",
        dest="lambda_crit",
        type=str,
        choices=["mad", "mad_update", "ut", "lut", "factor", "pcg", "eigval"],
        help="Criteria with which lambda is selected to estimate activity-inducing "
        "and innovation signals.",
        default="mad_update",
    )
    optional.add_argument(
        "-factor",
        "--factor",
        dest="factor",
        type=float,
        help="Factor to multiply noise estimate for lambda selection.",
        default=1,
    )
    optional.add_argument(
        "-jobs",
        "--jobs",
        dest="jobs",
        type=int,
        help="Number of cores to take to parallelize debiasing step (default = 4).",
        default=4,
    )
    optional.add_argument(
        "-lambda_echo",
        "--lambda_echo",
        dest="lambda_echo",
        type=int,
        help="Number of the TE data to calculate lambda for (default = last TE).",
        default=-1,
    )
    optional.add_argument(
        "-max_iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        help="Maximum number of iterations for FISTA (default = 100).",
        default=100,
    )
    optional.add_argument(
        "-min_iter",
        "--min_iter",
        dest="min_iter",
        type=int,
        help="Minimum number of iterations for FISTA (default = 10).",
        default=10,
    )
    optional.add_argument(
        "-stability",
        "--stability_selection",
        dest="do_stability_selection",
        action="store_true",
        help="Perform stability selection (default = False).",
        default=False,
    )
    optional.add_argument(
        "-debug",
        "--debug",
        dest="debug",
        action="store_true",
        help="Logs in the terminal will have increased "
        "verbosity, and will also be written into "
        "a .tsv file in the output directory.",
        default=False,
    )
    optional.add_argument(
        "-quiet",
        "--quiet",
        dest="quiet",
        help=argparse.SUPPRESS,
        action="store_true",
        default=False,
    )
    optional.add_argument(
        "-v", "--version", action="version", version=("%(prog)s " + __version__)
    )

    parser._action_groups.append(optional)

    return parser


if __name__ == "__main__":
    raise RuntimeError(
        "splora/cli/run.py should not be run directly;\n"
        "Please `pip install` splora and use the "
        "`splora` command"
    )
