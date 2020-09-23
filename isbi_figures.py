"""Script to generate figures with simulations for ISBI paper."""
import numpy as np
from low_rank_pfm.src.low_rank import low_rank
from low_rank_pfm.src.hrf_matrix import HRFMatrix


def perf_measure(y_actual, y_hat):
    """
    Measure performance.

    Calculate true positive, false positive, true negative and false negative.

    Parameters
    ----------
    y_actual : ndarray
        True or original time series.
    y_hat : ndarray
        Estimated time series.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


def main():
    """Generate figures for ISBI paper."""
    # Load data.
    data = np.load('sim_clean.npy')
    noise = np.load('sim_noise.npy')
    voxels_bold = 99

    # Repeat matrix to get up to 10000 voxels.
    noise = np.tile(noise, 10)
    data_clean = np.zeros(noise.shape)
    data_clean[:, :voxels_bold] = data[:, :voxels_bold]

    # List of percentages to generate different SNR and amount of voxels.
    noise_pcg_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    voxel_pcg_list = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Get number of time points and voxels.
    nscans = data_clean.shape[0]
    nvoxels = data_clean.shape[1]

    sensitivity = np.zeros((len(noise_pcg_list), len(voxel_pcg_list), voxels_bold))
    specificity = sensitivity.copy()
    accuracy = sensitivity.copy()
    L_error = np.zeros((len(noise_pcg_list), len(voxel_pcg_list)))
    nvoxels_mtx = L_error.copy()

    # Solve the Low Rank and Sparse regularization problem for different SNR.
    for noise_idx, noise_pcg in enumerate(noise_pcg_list):

        # Solve the Low Rank and Sparse regularization problem for different amount
        # of voxels with Low Rank components.
        for voxel_idx, voxel_pcg in enumerate(voxel_pcg_list):
            # Get data ready.
            nvoxels_iter = int(100 + (nvoxels - 100) * voxel_pcg)  # First 100 are not low rank.
            nvoxels_mtx[noise_idx, voxel_idx] = nvoxels_iter
            noise_iter = noise[:, :nvoxels_iter] * noise_pcg
            data_iter = data_clean[:, :nvoxels_iter] + noise_iter

            breakpoint()

            # Generate design matrix with shiftted HRF.
            hrf_obj = HRFMatrix(TR=2, nscans=nscans, TE=[0], has_integrator=False)
            hrf_norm = hrf_obj.generate_hrf().X_hrf_norm

            # Solve Low Rank and Sparse regularization problem.
            L, S, eigvec = low_rank(data=data_iter, hrf=hrf_norm)

            # Calculate specificity, sensitivity and accuracy for betas.
            TP = np.zeros((voxels_bold))
            FP = np.zeros((voxels_bold))
            TN = np.zeros((voxels_bold))
            FN = np.zeros((voxels_bold))

            for i in range(voxels_bold):
                TP[i], FP[i], TN[i], FN[i] = perf_measure(data_clean[:, i] != 0, S[:, i] != 0)

            sensitivity[noise_idx, voxel_idx, :] = TP / (TP + FN)
            specificity[noise_idx, voxel_idx, :] = TN / (TN + FP)
            accuracy[noise_idx, voxel_idx, :] = (TP + TN) / (TP + TN + FP + FN)

            # Calculate error for low rank component.
            L_error[noise_idx, voxel_idx] = (np.linalg.norm(L - noise_iter, ord='fro')
                                             / np.linalg.norm(noise_iter, ord='fro'))

    return sensitivity, specificity, accuracy, L_error, nvoxels_mtx


if __name__ == '__main__':
    main()
