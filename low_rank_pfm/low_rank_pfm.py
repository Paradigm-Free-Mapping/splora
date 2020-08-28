import sys

import numpy as np
import scipy as sci
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img, new_img_like

from low_rank_pfm.cli.run import _get_parser
from low_rank_pfm.src.low_rank import low_rank
from low_rank_pfm.src.hrf_matrix import HRFMatrix


def debiasing(x, y, beta):

    beta_out = np.zeros(beta.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(beta) > 10 * np.finfo(float).eps)[1])

    print('Debiasing results...')
    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > 10 * np.finfo(float).eps)[0]

        X_events = x[:, index_events_opt]
        beta2save = np.zeros((beta.shape[0], 1))

        coef_LSfitdebias, residuals, rank, s = sci.linalg.lstsq(
            X_events, y[:, index_voxels[voxidx]], cond=None)
        beta2save[index_events_opt, 0] = coef_LSfitdebias

        beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))
        fitts_out[:, index_voxels[voxidx]] = np.dot(X_events, coef_LSfitdebias)

    print('Debiasing completed.')
    return(beta_out, fitts_out)


def low_rank_pfm(data_filename, mask_filename, tr, te=[0]):

    data_img = load_img(data_filename, dtype='float32')
    masker = NiftiMasker(mask_img=mask_filename, standardize=False)
    data_masked = masker.fit_transform(data_filename)

    hrf_obj = HRFMatrix(TR=tr, nscans=int(data_masked.shape[0]), TE=te)
    hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

    L, S = low_rank(data=data_masked, hrf=hrf_norm)

    # Debiasing
    S_deb, S_fitts = debiasing(x=hrf_norm, y=data_masked, beta=S)

    masker.inverse_transform(L).to_filename('L.nii.gz')
    # nii = new_img_like(ref_niimg=data_img, data=L_img, copy_header=True)
    # nii.set_data_dtype(L_img.dtype)
    # nii.to_filename('L.nii.gz')

    masker.inverse_transform(S_deb).to_filename('S.nii.gz')
    # nii = new_img_like(ref_niimg=data_img, data=S_img, copy_header=True)
    # nii.set_data_dtype(S_img.dtype)
    # nii.to_filename('S.nii.gz')

    masker.inverse_transform(S_fitts).to_filename('fitts.nii.gz')
    # nii = new_img_like(ref_niimg=data_img, data=S_fitts_img, copy_header=True)
    # nii.set_data_dtype(S_fitts_img.dtype)
    # nii.to_filename('fitts.nii.gz')


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    low_rank_pfm(**vars(options))


if __name__ == '__main__':
    _main(sys.argv[1:])
