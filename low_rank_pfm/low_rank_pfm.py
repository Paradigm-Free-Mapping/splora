"""Main."""
import sys

import nibabel as nib
import numpy as np
import scipy as sci

from low_rank_pfm.cli.run import _get_parser
from low_rank_pfm.src.low_rank import low_rank
from low_rank_pfm.src.hrf_matrix import HRFMatrix
from low_rank_pfm.io import read_data, reshape_data, update_history


def debiasing(x, y, beta, thr=1e-3):
    """
    Debias beta estimates.

    Args:
        x ([type]): [description]
        y ([type]): [description]
        beta ([type]): [description]
        thr ([type], optional): [description]. Defaults to 1e-3.
    """
    beta_out = np.zeros(beta.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(beta) > thr)[1])

    print('Debiasing results...')
    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > thr)[0]

        if index_events_opt.size != 0:
            X_events = x[:, index_events_opt]
            beta2save = np.zeros((beta.shape[0], 1))

            coef_LSfitdebias, _, _, _ = sci.linalg.lstsq(
                X_events, y[:, index_voxels[voxidx]], cond=None)
            beta2save[index_events_opt, 0] = coef_LSfitdebias

            beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))
            fitts_out[:, index_voxels[voxidx]] = np.dot(X_events, coef_LSfitdebias)
        else:
            beta_out[:, index_voxels[voxidx]] = np.zeros((beta.shape[0], 1))
            fitts_out[:, index_voxels[voxidx]] = np.zeros((beta.shape[0], 1))

    print('Debiasing completed.')
    return(beta_out, fitts_out)


def low_rank_pfm(data_filename, mask_filename, output_filename, tr, te=[0], thr=1e-3,
                 lambda_weight=1.1, group=0):
    """
    Low-rank PFM main function.

    Args:
        data_filename ([type]): [description]
        mask_filename ([type]): [description]
        output_filename ([type]): [description]
        tr ([type]): [description]
        te (list, optional): [description]. Defaults to [0].
        thr ([type], optional): [description]. Defaults to 1e-3.
    """
    te_str = str(te).strip('[]')
    arguments = f'-i {data_filename} -m {mask_filename} -o {output_filename} -tr {tr} '
    arguments += f'-te {te_str} -thr {thr} -l {lambda_weight}'
    command_str = f'low_rank_pfm {arguments}'

    print('Reading data...')
    data_masked, data_header, dims, mask_idxs = read_data(data_filename, mask_filename)
    print('Data read.')

    hrf_obj = HRFMatrix(TR=tr, nscans=int(data_masked.shape[0]), TE=te, has_integrator=False)
    hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

    L, S, eigen_vecs = low_rank(data=data_masked, hrf=hrf_norm, lambda_weight=lambda_weight,
                                group=group)

    # Debiasing
    S_deb, S_fitts = debiasing(x=hrf_norm, y=data_masked, beta=S, thr=thr)

    print('Saving results...')
    # Save estimated fluctuations
    L_reshaped = reshape_data(L, dims, mask_idxs)
    L_nib = nib.Nifti1Image(L_reshaped, None, header=data_header)
    L_output_filename = f'{output_filename}_fluc.nii.gz'
    L_nib.to_filename(L_output_filename)

    S_reshaped = reshape_data(S_deb, dims, mask_idxs)
    S_nib = nib.Nifti1Image(S_reshaped, None, header=data_header)
    S_output_filename = f'{output_filename}_beta.nii.gz'
    S_nib.to_filename(S_output_filename)

    S_fitts_reshaped = reshape_data(S_fitts, dims, mask_idxs)
    S_fitts_nib = nib.Nifti1Image(S_fitts_reshaped, None, header=data_header)
    S_fitts_output_filename = f'{output_filename}_fitts.nii.gz'
    S_fitts_nib.to_filename(S_fitts_output_filename)

    # Saving eigen vectors
    for i in range(eigen_vecs.shape[1]):
        eigen_vecs_output_filename = f'{output_filename}_eigenvec_{i+1}.1D'
        np.savetxt(eigen_vecs_output_filename, np.squeeze(eigen_vecs[:, i]))

    print('Results saved.')

    print('Updating file history...')
    update_history(L_output_filename, command_str)
    update_history(S_output_filename, command_str)
    update_history(S_fitts_output_filename, command_str)
    print('File history updated.')

    print('Low-Rank and Sparse PFM finished.')


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    low_rank_pfm(**vars(options))


if __name__ == '__main__':
    _main(sys.argv[1:])
