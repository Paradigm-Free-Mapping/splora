"""Main."""
import sys

import nibabel as nib
import numpy as np

from low_rank_pfm.cli.run import _get_parser
from low_rank_pfm.src.debiasing import debiasing_spike, debiasing_block
from low_rank_pfm.io import read_data, reshape_data, update_history
from low_rank_pfm.src.fista import fista
from low_rank_pfm.src.hrf_matrix import HRFMatrix


def low_rank_pfm(
    data_filename,
    mask_filename,
    output_filename,
    tr,
    te=[0],
    thr=1e-3,
    eigthr=0.25,
    lambda_weight=1.1,
    group=0,
    do_debias=False,
    is_pfm=False,
    lambda_crit="mad_update",
    factor=1,
):
    """Low-rank PFM main function.

    Parameters
    ----------
    data_filename : [type]
        [description]
    mask_filename : [type]
        [description]
    output_filename : [type]
        [description]
    tr : [type]
        [description]
    te : list, optional
        [description], by default [0]
    thr : [type], optional
        [description], by default 1e-3
    eigthr : float, optional
        [description], by default 0.25
    lambda_weight : float, optional
        [description], by default 1.1
    group : int, optional
        [description], by default 0
    do_debias : bool, optional
        [description], by default False
    is_pfm : bool, optional
        [description], by default False
    lambda_crit : str, optional
        [description], by default "mad_update"
    factor : int, optional
        [description], by default 1
    """
    te_str = str(te).strip("[]")
    arguments = f"-i {data_filename} -m {mask_filename} -o {output_filename} -tr {tr} "
    arguments += f"-te {te_str} -thr {thr} -l {lambda_weight}"
    command_str = f"low_rank_pfm {arguments}"

    n_te = len(te)

    print("Reading data...")
    if n_te == 1:
        data_masked, data_header, dims, mask_idxs = read_data(
            data_filename[0], mask_filename
        )
        nscans = data_masked.shape[0]
    elif n_te > 1:
        for te_idx in range(n_te):
            data_temp, data_header, dims, mask_idxs = read_data(
                data_filename[te_idx], mask_filename
            )
            if te_idx == 0:
                data_masked = data_temp
                nscans = data_temp.shape[0]
            else:
                data_masked = np.concatenate((data_masked, data_temp), axis=0)

            print(f"{te_idx + 1}/{n_te} echoes...")

    print("Data read.")

    hrf_obj = HRFMatrix(TR=tr, nscans=nscans, TE=te, has_integrator=False)
    hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

    S, L, eigen_vecs, eigen_maps = fista(
        X=hrf_norm,
        y=data_masked,
        nscans=nscans,
        n_te=n_te,
        lambda_crit=lambda_crit,
        factor=factor,
    )

    # Debiasing
    if do_debias:
        S_deb, S_fitts = debiasing_spike(x=hrf_norm, y=data_masked, beta=S)
    else:
        S_deb = S
        S_fitts = np.dot(hrf_norm, S_deb)

    print("Saving results...")
    # Save estimated fluctuations
    L_reshaped = reshape_data(L, dims, mask_idxs)
    L_nib = nib.Nifti1Image(L_reshaped, None, header=data_header)
    # L_nib = new_nii_like(data_filename, L_reshaped)
    L_output_filename = f"{output_filename}_fluc.nii.gz"
    L_nib.to_filename(L_output_filename)
    update_history(L_output_filename, command_str)

    S_reshaped = reshape_data(S_deb, dims, mask_idxs)
    S_nib = nib.Nifti1Image(S_reshaped, None, header=data_header)
    # S_nib = new_nii_like(data_filename, S_reshaped)
    S_output_filename = f"{output_filename}_beta.nii.gz"
    S_nib.to_filename(S_output_filename)
    update_history(S_output_filename, command_str)

    if n_te == 1:
        S_fitts_reshaped = reshape_data(S_fitts, dims, mask_idxs)
        S_fitts_nib = nib.Nifti1Image(S_fitts_reshaped, None, header=data_header)
        # S_fitts_nib = new_nii_like(data_filename, S_fitts_reshaped)
        S_fitts_output_filename = f"{output_filename}_fitts.nii.gz"
        S_fitts_nib.to_filename(S_fitts_output_filename)
        update_history(S_fitts_output_filename, command_str)
    elif n_te > 1:
        for te_idx in range(n_te):
            te_data = S_fitts[te_idx * nscans : (te_idx + 1) * nscans, :]
            S_fitts_reshaped = reshape_data(
                te_data, dims, mask_idxs
            )
            S_fitts_nib = nib.Nifti1Image(S_fitts_reshaped, None, header=data_header)
            # S_fitts_nib = new_nii_like(data_filename, S_fitts_reshaped)
            S_fitts_output_filename = f"{output_filename}_fitts_E0{te_idx + 1}.nii.gz"
            S_fitts_nib.to_filename(S_fitts_output_filename)
            update_history(S_fitts_output_filename, command_str)

    if is_pfm is False:
        # Saving eigen vectors and maps
        for i in range(eigen_vecs.shape[1]):
            eigen_vecs_output_filename = f"{output_filename}_eigenvec_{i+1}.1D"
            np.savetxt(eigen_vecs_output_filename, np.squeeze(eigen_vecs[:, i]))
            eigen_map_reshaped = reshape_data(
                np.expand_dims(eigen_maps[i, :], axis=0), dims, mask_idxs
            )
            eigen_map_nib = nib.Nifti1Image(
                eigen_map_reshaped, None, header=data_header
            )
            # eigen_map_nib = new_nii_like(data_filename, eigen_map_reshaped)
            eigen_map_output_filename = f"{output_filename}_eigenmap_{i+1}.nii.gz"
            eigen_map_nib.to_filename(eigen_map_output_filename)

    print("Results saved.")
    print("Low-Rank and Sparse PFM finished.")


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    low_rank_pfm(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
