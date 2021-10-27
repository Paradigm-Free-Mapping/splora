"""Main."""
import datetime
import logging
import os
import sys
from os import path as op

import numpy as np

from splora import utils
from splora.cli.run import _get_parser
from splora.deconvolution import fista
from splora.deconvolution.debiasing import debiasing_block, debiasing_spike
from splora.deconvolution.hrf_matrix import HRFMatrix
from splora.io import read_data, write_data

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def splora(
    data_filename,
    mask_filename,
    output_filename,
    tr,
    out_dir,
    te=[0],
    eigthr=0.1,
    group=0,
    do_debias=False,
    pfm_only=False,
    lambda_crit="mad_update",
    factor=1,
    block_model=False,
    jobs=4,
    lambda_echo=-1,
    debug=False,
    quiet=False,
):
    """Run main workflow of splora.

    Parameters
    ----------
    data_filename : list of str or path
        Paths to input filenames.
    mask_filename : str or path
        Path to mask file.
    output_filename : str or path
        Base filename for all splora outputs.
    tr : float
        TR of the acquisition.
    out_dir : str or path
        Path to output directory.
    te : list, optional
        TE of the acquisition in ms, by default [0]
    eigthr : float, optional
        Minimum percentage gap between the eigen values of selected low-rank components,
        by default 0.1
    group : float, optional
        Weight for grouping effect over sparsity, by default 0
    do_debias : bool, optional
        Whether to perform the debiasing step, by default False
    pfm_only : bool, optional
        Whether to run without the low-rank model, by default False
    lambda_crit : str, optional
        Criteria to select regularization parameter lambda, by default "mad_update"
    factor : int, optional
        Factor by which to multiply the value of lambda, by default 1
        Only used when "factor" criteria is selected.
    block_model : bool, optional
        Whether to use the block model in favor of the spike model, by default False
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/LGRing of messages. Default is False.
    """
    data_str = str(data_filename).strip("[]")
    te_str = str(te).strip("[]")
    arguments = f"-i {data_str} -m {mask_filename} -o {output_filename} -tr {tr} "
    arguments += f"-d {out_dir} -te {te_str} -eigthr {eigthr} -group {group} -crit {lambda_crit} "
    arguments += f"-factor {factor} "
    if do_debias:
        arguments += "--debias "
    if pfm_only:
        arguments += "-pfm "
    if block_model:
        arguments += "-block "
    if debug:
        arguments += "-debug "
    if quiet:
        arguments += "-quiet"
    command_str = f"splora {arguments}"

    # Generate output directory if it doesn't exist
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Save command into sh file
    command_file = open(os.path.join(out_dir, "call.sh"), "w")
    command_file.write(command_str)
    command_file.close()

    LGR = logging.getLogger("GENERAL")
    # RefLGR = logging.getLogger("REFERENCES")
    # create logfile name
    basename = "splora_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    refname = op.join(out_dir, "_references.txt")
    utils.setup_loggers(logname, refname, quiet=quiet, debug=debug)

    # Main references
    RefLGR.info(
        "Uruñuela, E., Moia, S., & Caballero-Gaudes, C. (2021, April). A Low Rank "
        "and Sparse Paradigm Free Mapping Algorithm For Deconvolution of FMRI Data. "
        "In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) "
        "(pp. 1726-1729). IEEE."
    )

    LGR.info("Using output directory: {}".format(out_dir))

    n_te = len(te)

    if all(i >= 1 for i in te):
        te = [x / 1000 for x in te]

    LGR.info("Reading data...")
    if n_te == 1:
        data_masked, data_header, mask_img = read_data(data_filename[0], mask_filename)
        nscans = data_masked.shape[0]
    elif n_te > 1:
        for te_idx in range(n_te):
            data_temp, data_header, mask_img = read_data(data_filename[te_idx], mask_filename)
            if te_idx == 0:
                data_masked = data_temp
                nscans = data_temp.shape[0]
            else:
                # data_temp, _, _, _ = read_data(data_filename[te_idx], mask_filename, mask_idxs)
                data_masked = np.concatenate((data_masked, data_temp), axis=0)

            LGR.info(f"{te_idx + 1}/{n_te} echoes...")

    LGR.info("Data read.")

    hrf_obj = HRFMatrix(TR=tr, nscans=nscans, TE=te, block=block_model)
    hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

    S, eigen_vecs, eigen_maps, noise_estimate, lambda_val = fista.fista(
        hrf=hrf_norm,
        y=data_masked,
        n_te=n_te,
        lambda_crit=lambda_crit,
        factor=factor,
        eigen_thr=eigthr,
        group=group,
        pfm_only=pfm_only,
        block_model=block_model,
        tr=tr,
        te=te,
        jobs=jobs,
        lambda_echo=lambda_echo,
    )

    # Debiasing
    if do_debias:
        if block_model:
            hrf_obj = HRFMatrix(TR=tr, nscans=nscans, TE=te, block=False)
            hrf_norm = hrf_obj.generate_hrf().X_hrf_norm
            S_deb = debiasing_block(hrf=hrf_norm, y=data_masked, auc=S)
            S_fitts = np.dot(hrf_norm, S_deb)
        else:
            S_deb, S_fitts = debiasing_spike(hrf=hrf_norm, y=data_masked, auc=S)
    else:
        S_deb = S
        S_fitts = np.dot(hrf_norm, S_deb)

    LGR.info("Saving results...")
    # Save innovation signal
    if block_model:
        output_name = f"{output_filename}_innovation.nii.gz"
        write_data(S, os.path.join(out_dir, output_name), mask_img, data_header, command_str)

        if not do_debias:
            hrf_obj = HRFMatrix(TR=tr, nscans=nscans, TE=te, block=False)
            hrf_norm = hrf_obj.generate_hrf().X_hrf_norm
            S_deb = np.dot(np.tril(np.ones(nscans)), S_deb)
            S_fitts = np.dot(hrf_norm, S_deb)

    # Save activity-inducing signal
    output_name = f"{output_filename}_beta.nii.gz"
    write_data(S_deb, os.path.join(out_dir, output_name), mask_img, data_header, command_str)

    if n_te == 1:
        output_name = f"{output_filename}_fitts.nii.gz"
        write_data(
            S_fitts,
            os.path.join(out_dir, output_name),
            mask_img,
            data_header,
            command_str,
        )
    elif n_te > 1:
        for te_idx in range(n_te):
            te_data = S_fitts[te_idx * nscans : (te_idx + 1) * nscans, :]
            output_name = f"{output_filename}_fitts_E0{te_idx + 1}.nii.gz"
            write_data(
                te_data,
                os.path.join(out_dir, output_name),
                mask_img,
                data_header,
                command_str,
            )

    if pfm_only is False:
        # Saving eigen vectors and maps
        for i in range(eigen_vecs.shape[1]):
            for te_idx in range(n_te):
                # Time-series
                output_name = f"{output_filename}_eigenvec_{i+1}_E0{te_idx + 1}.1D"
                if te_idx == 0:
                    eig_echo = eigen_vecs[:nscans, i]
                else:
                    eig_echo = eigen_vecs[te_idx * nscans : (te_idx + 1) * nscans, i]
                np.savetxt(os.path.join(out_dir, output_name), np.squeeze(eig_echo))
            # Maps
            low_rank_map = np.expand_dims(eigen_maps[i, :], axis=0)
            output_name = f"{output_filename}_eigenmap_{i+1}.nii.gz"
            write_data(
                low_rank_map,
                os.path.join(out_dir, output_name),
                mask_img,
                data_header,
                command_str,
            )

    # Save noise estimate
    for te_idx in range(n_te):
        output_name = f"{output_filename}_MAD_E0{te_idx + 1}.nii.gz"
        if te_idx == 0:
            y_echo = data_masked[:nscans, :]
        else:
            y_echo = data_masked[te_idx * nscans : (te_idx + 1) * nscans, :]
        _, _, noise_estimate = fista.select_lambda(hrf=hrf_norm, y=y_echo)
        write_data(
            np.expand_dims(noise_estimate, axis=0),
            os.path.join(out_dir, output_name),
            mask_img,
            data_header,
            command_str,
        )

    # Save lambda
    output_name = f"{output_filename}_lambda.nii.gz"
    write_data(
        np.expand_dims(lambda_val, axis=0),
        os.path.join(out_dir, output_name),
        mask_img,
        data_header,
        command_str,
    )

    LGR.info("Results saved.")

    LGR.info("Low-Rank and Sparse PFM finished.")
    utils.teardown_loggers()


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    splora(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
