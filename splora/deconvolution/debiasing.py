"""Debiasing functions for PFM."""
import logging

import numpy as np
import scipy as sci
from scipy.signal import find_peaks
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from joblib import Parallel, delayed

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


# Performs the debiasing step on an AUC timeseries obtained considering the integrator model
def innovation_to_block(hrf, y, auc, is_ls):
    """Perform debiasing with the block model.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    auc : (T x S) array_like
        Matrix containing the non-zero coefficients selected as neuronal-related.
    is_ls : bool
        Whether least squares is solved in favor of ridge regression.

    Returns
    -------
    beta : (T x S) array_like
        Debiased activity-inducing signal obtained from estimated innovation signal.
    S : (T x L) array_like
        Transformation matrix used to integrate the innovation signal into activity-inducing
        signal. L stands for the number of steps to integrate.
    """
    # Find indexes of nonzero coefficients
    nonzero_idxs = np.where(auc != 0)[0]
    n_nonzero = len(nonzero_idxs)  # Number of nonzero coefficients

    # Initiates beta
    beta = np.zeros((auc.shape))
    S = 0

    if n_nonzero != 0:
        # Initiates matrix S and array of labels
        S = np.zeros((hrf.shape[1], n_nonzero + 1))
        labels = np.zeros((auc.shape[0]))

        # Gives values to S design matrix based on nonzeros in AUC
        # It also stores the labels of the changes in the design matrix
        # to later generate the debiased timeseries with the obtained betas
        for idx in range(n_nonzero + 1):
            if idx == 0:
                S[0 : nonzero_idxs[idx], idx] = 1
                labels[0 : nonzero_idxs[idx]] = idx
            elif idx == n_nonzero:
                S[nonzero_idxs[idx - 1] :, idx] = 1
                labels[nonzero_idxs[idx - 1] :] = idx
            else:
                S[nonzero_idxs[idx - 1] : nonzero_idxs[idx], idx] = 1
                labels[nonzero_idxs[idx - 1] : nonzero_idxs[idx]] = idx

        # Performs the least squares to obtain the beta amplitudes
        if is_ls:
            beta_amplitudes, _, _, _ = np.linalg.lstsq(
                np.dot(hrf, S), y, rcond=None
            )  # b-ax --> returns x
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]).fit(np.dot(hrf, S), y)
            beta_amplitudes = clf.coef_

        # Positions beta amplitudes in the entire timeseries
        for amp_change in range(n_nonzero + 1):
            beta[labels == amp_change] = beta_amplitudes[amp_change]

    return beta, S


def do_debias_block(hrf, y, auc, dist=2):
    """Perform debiasing with the block model.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x 1) array_like
        Matrix with fMRI data provided to splora in a voxel.
    auc : (T x 1) array_like
        Matrix containing the non-zero coefficients selected as neuronal-related in a voxel.
    dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 2

    Returns
    -------
    beta_out : array_like
        Debiased activity-inducing signal obtained from estimated innovation signal in a voxel.
    """
    # Keep only maximum values in AUC peaks
    temp = np.zeros((auc.shape[0],))
    peak_idxs, _ = find_peaks(abs(auc), distance=dist)
    temp[peak_idxs] = auc[peak_idxs].copy()

    auc = temp.copy()

    beta_out, _ = innovation_to_block(hrf, y, auc, is_ls=True)

    return beta_out


def debiasing_block(hrf, y, auc, dist=2):
    """Voxelwise block model debiasing workflow.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    auc : (T x S) array_like
        Matrix containing the non-zero coefficients selected as neuronal-related.
    dist : int, optional
        Minimum number of TRs in between of the peaks found, by default 2

    Returns
    -------
    beta_out : array_like
        Debiased activity-inducing signal obtained from estimated innovation signal.
    """
    nscans = auc.shape[0]
    nvoxels = auc.shape[1]

    # Initiates beta matrix
    beta_out = np.zeros((nscans, nvoxels))

    LGR.info("Starting debiasing step...")
    # Performs debiasing
    debiased = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(do_debias_block)(hrf, y[:, voxidx], auc[:, voxidx])
        for voxidx in tqdm(range(nvoxels))
    )
    for vox_idx in range(nvoxels):
        beta_out[:, vox_idx] = debiased[vox_idx]

    LGR.info("Debiasing step finished")
    return beta_out


def do_debias_spike(hrf, y, auc):
    """Perform debiasing with the spike model.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x 1) array_like
        Array with fMRI data of a voxel provided to splora.
    auc : (T x 1) array_like
        Array containing the non-zero coefficients selected as neuronal-related.

    Returns
    -------
    beta_out : array_like
        Debiased activity-inducing signal in a voxel.
    fitts_out : array_like
        Debiased activity-inducing signal convolved with the HRF in a voxel.
    """
    index_events_opt = np.where(abs(auc) > 10 * np.finfo(float).eps)[0]
    beta2save = np.zeros((auc.shape[0], 1))

    hrf_events = hrf[:, index_events_opt]

    coef_LSfitdebias, _, _, _ = sci.linalg.lstsq(hrf_events, y, cond=None)
    beta2save[index_events_opt, 0] = coef_LSfitdebias
    fitts_out = np.squeeze(np.dot(hrf, beta2save))
    beta_out = beta2save.reshape(len(beta2save))

    return beta_out, fitts_out


def debiasing_spike(hrf, y, auc):
    """Perform voxelwise debiasing with spike model.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    auc : (T x S) array_like
        Matrix containing the non-zero coefficients selected as neuronal-related.

    Returns
    -------
    beta_out : array_like
        Debiased activity-inducing signal.
    fitts_out : array_like
        Debiased activity-inducing signal convolved with the HRF.
    """
    beta_out = np.zeros(auc.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(auc) > 10 * np.finfo(float).eps)[1])

    LGR.info("Performing debiasing step...")
    debiased = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(do_debias_spike)(hrf, y[:, index_voxels[voxidx]], auc[:, index_voxels[voxidx]])
        for voxidx in tqdm(range(len(index_voxels)))
    )

    for voxidx in range(len(index_voxels)):
        beta_out[:, index_voxels[voxidx]] = debiased[voxidx][0]
        fitts_out[:, index_voxels[voxidx]] = debiased[voxidx][1]

    LGR.info("Debiasing step finished")
    return beta_out, fitts_out
