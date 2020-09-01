import sys

import nibabel as nib
import numpy as np
import scipy as sci
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from nilearn.image import load_img, new_img_like

from low_rank_pfm.cli.run import _get_parser
from low_rank_pfm.src.low_rank import low_rank
from low_rank_pfm.src.hrf_matrix import HRFMatrix
from low_rank_pfm.io import read_data, reshape_data


def debiasing(x, y, beta):

    beta_out = np.zeros(beta.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(beta) > 1e-2)[1])

    print('Debiasing results...')
    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > 1e-2)[0]

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


def low_rank_pfm(data_filename, mask_filename, output_filename, tr, te=[0]):

    # data_img = load_img(data_filename, dtype='float32')
    # mask_img = load_img(mask_filename, dtype='float32')
    # data_masked = apply_mask(data_img, mask_img)
    print('Reading data...')
    orig_data_img = nib.load(data_filename)
    data_masked = apply_mask(data_filename, mask_filename)
    # data_masked, data_header, dims, mask_idxs = read_data(data_filename, mask_filename)
    print('Data read.')
    # masker = NiftiMasker(mask_img=mask_filename, standardize=False)
    # data_masked = masker.fit_transform(data_filename)
    # masker.t_r = tr

    hrf_obj = HRFMatrix(TR=tr, nscans=int(data_masked.shape[0]), TE=te, has_integrator=False)
    hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

    L, S = low_rank(data=data_masked, hrf=hrf_norm)

    # Debiasing
    S_deb, S_fitts = debiasing(x=hrf_norm, y=data_masked, beta=S)

    breakpoint()

    print('Saving results...')

    # Save estimated fluctuations
    # masker.inverse_transform(L).to_filename(f'{output_filename}_fluc.nii.gz')
    # L_img = unmask(L, mask_img)
    # L_img.header.from_header(header=data_img.header)
    # nii = new_img_like(ref_niimg=data_img, data=L_img, copy_header=True)
    # L_reshaped = reshape_data(L, dims, mask_idxs)
    # L_nib = nib.Nifti1Image(L_reshaped, None, header=data_header)
    # L_nii = new_nii_like(ref_img=data_img, data=L, affine=data_img.affine)
    # L_nib.to_filename(f'{output_filename}_fluc.nii.gz')
    L_img = unmask(L, mask_filename)
    L_new_img = nib.Nifti1Image(L_img.get_fdata(), L_img.affine, header=orig_data_img.header)
    L_new_img.to_filename(f'{output_filename}_fluc.nii.gz')

    # masker.inverse_transform(S_deb).to_filename(f'{output_filename}_beta.nii.gz')
    # S_img = unmask(S_deb, mask_img)
    # S_img.header.from_header(header=data_img.header)
    # nii = new_img_like(ref_niimg=data_img, data=S_img, copy_header=True)
    # S_reshaped = reshape_data(S, dims, mask_idxs)
    # S_nib = nib.Nifti1Image(S_reshaped, None, header=data_header)
    # S_nii = new_nii_like(ref_img=data_img, data=S_deb, affine=data_img.affine)
    # S_nib.to_filename(f'{output_filename}_beta.nii.gz')
    S_img = unmask(S_deb, mask_filename)
    S_new_img = nib.Nifti1Image(S_img.get_fdata(), S_img.affine, header=orig_data_img.header)
    S_new_img.to_filename(f'{output_filename}_beta.nii.gz')

    # masker.inverse_transform(S_fitts).to_filename(f'{output_filename}_fitts.nii.gz')
    # S_fitts_img = unmask(S_fitts, mask_img)
    # S_fitts_img.header.from_header(header=data_img.header)
    # nii = new_img_like(ref_niimg=data_img, data=S_fitts_img, copy_header=True)
    # S_fitts_reshaped = reshape_data(S_fitts, dims, mask_idxs)
    # S_fitts_nib = nib.Nifti1Image(S_fitts_reshaped, None, header=data_header)
    # S_fitts_nii = new_nii_like(ref_img=data_img, data=S_fitts, affine=data_img.affine)
    # S_fitts_nib.to_filename(f'{output_filename}_fitts.nii.gz')
    Fitts_img = unmask(S_fitts, mask_filename)
    Fitts_new_img = nib.Nifti1Image(Fitts_img.get_fdata(), Fitts_img.affine, header=orig_data_img.header)
    Fitts_new_img.to_filename(f'{output_filename}_fitts.nii.gz')

    print('Results saved.')
    print('Low-Rank and Sparse PFM finished.')


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    low_rank_pfm(**vars(options))


if __name__ == '__main__':
    _main(sys.argv[1:])
