"""FISTA solver for Low-Rank and Sparse PFM."""

import logging

import numpy as np
from pySPFM.deconvolution.debiasing import debiasing_block, debiasing_spike
from pySPFM.deconvolution.hrf_generator import HRFMatrix
from pySPFM.deconvolution.select_lambda import select_lambda
from scipy import linalg

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def proximal_operator_lasso(y, thr):
    """Perform soft-thresholding.

    Parameters
    ----------
    y : array_like
        Input data to be soft-thresholded.
    thr : float
        Thresholding value.

    Returns
    -------
    x : array_like
        Soft-thresholded data.
    """
    x = y * np.maximum(np.zeros(y.shape), 1 - (thr / abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return x


def proximal_operator_mixed_norm(y, thr, rho_val=0.8, groups="space"):
    """Apply proximal operator for L2,1 + L1 mixed-norm.

    Parameters
    ----------
    y : array_like
        Input data to be soft-thresholded.
    thr : float
        Thresholding value.
    rho_val : float, optional
        Weight for sparsity over grouping effect, by default 0.8
    groups : str, optional
        Dimension to apply grouping on, by default "space"

    Returns
    -------
    x : array_like
        Data thresholded with L2,1 + L1 mixed-norm proximal operator.
    """
    # Division parameter of proximal operator
    div = np.nan_to_num(y / np.abs(y))

    # First parameter of proximal operator
    p_one = np.maximum(np.zeros(y.shape), (np.abs(y) - thr * rho_val))

    # Second parameter of proximal operator
    if groups == "space":
        foo = np.sum(
            np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=1
        )
        foo = foo.reshape(len(foo), 1)
        foo = np.dot(foo, np.ones((1, y.shape[1])))
    else:
        foo = np.sum(
            np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=0
        )
        foo = foo.reshape(1, len(foo))
        foo = np.dot(np.ones((y.shape[0], 1), foo))

    p_two = np.maximum(
        np.zeros(y.shape),
        np.ones(y.shape) - np.nan_to_num(thr * (1 - rho_val) / np.sqrt(foo)),
    )

    # Proximal operation
    x = div * p_one * p_two

    # Return result
    return x


def fista(
    hrf,
    y,
    n_te,
    lambd=None,
    max_iter=100,
    min_iter=10,
    lambda_crit="mad_update",
    precision=None,
    eigen_thr=0.1,
    tol=1e-6,
    factor=1,
    group=0.2,
    pfm_only=False,
    out_dir=None,
    block_model=False,
    tr=2,
    te=None,
    jobs=4,
    lambda_echo=-1,
):
    """Solve inverse problem with FISTA.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    n_te : int
        Number of echo-times provided.
    lambd : array_like, optional
        Regularization parameter lambda for each voxel, by default None.
        If None, lambda is selected based on lambda_crit.
    max_iter : int, optional
        Maximum number of iterations for FISTA, by default 100
    min_iter : int, optional
        Minimum number of iterations for FISTA, by default 10
    lambda_crit : str, optional
        Criteria to select the regularization parameter lambda, by default "mad_update"
    precision : float, optional
        Minimum value with which lambda is considered to have converged to the MAD estimate
        of the noise, by default None
    eigen_thr : float, optional
        Minimum percentage gap between the eigen values of selected low-rank components,
        by default 0.1
    tol : float, optional
        Value to which FISTA is considered to have converged, by default 1e-6
    factor : int, optional
        Factor by which the regularization parameter lambda is multiplied, by default 1
    group : float, optional
        Weight for grouping effect over sparsity, by default 0.2
    pfm_only : boolean, optional
        Whether PFM is run with original formulation, i.e., no low-rank, by default False
    te : list, optional
        List of echo times in seconds, by default [1].

    Returns
    -------
    S : (T x S) aray_like
        Estimated activity-inducing signal (for spike model) or innovation signal
        (for block model).
    eig_vecs : (T x ) aray_like
        Time-series of the estimated low-rank components.
    eig_maps : (S x ) aray_like
        Spatial maps of the estimated low-rank components.
    """
    if te is None:
        te = [1]
    nvoxels = y.shape[1]
    nscans = hrf.shape[1]

    c_ist = 1 / (linalg.norm(hrf) ** 2)
    hrf_trans = hrf.T
    hrf_cov = np.dot(hrf_trans, hrf)
    v = np.dot(hrf_trans, y)

    y_fista_S = np.zeros((nscans, nvoxels), dtype=np.float32)
    y_fista_L = np.zeros(y.shape)
    S_fitts = y_fista_L.copy()
    y_fista_A = y_fista_L.copy()
    S = y_fista_S.copy()
    # L lives in observation space (same shape as y)
    # This allows FISTA to work with subsampled data
    L = np.zeros(y.shape)
    A = y.copy()

    keep_idx = 0
    t_fista = 1
    update_lambda = False

    # Select lambda for each voxel based on criteria if no lambda is given
    if lambd is None:
        lambda_S, update_lambda, noise_estimate = select_lambda(
            hrf, y, factor=factor, criterion=lambda_crit, lambda_echo=lambda_echo
        )
    else:
        lambda_S = lambd
        noise_estimate = np.zeros(nvoxels)

    if precision is None and update_lambda:
        precision = noise_estimate / 100000

    # Perform FISTA
    for num_iter in range(max_iter):
        LGR.info(f"Iteration {num_iter + 1}/{max_iter}")

        if not pfm_only:
            if block_model:
                data_fidelity = y - S_fitts - y_fista_L
            else:
                data_fidelity = A.copy() - S_fitts

        # Save results from previous iteration
        S_old = S.copy()
        L_old = L.copy()
        A_old = A.copy()
        y_ista_S = y_fista_S.copy()
        y_ista_L = y_fista_L.copy()
        y_ista_A = y_fista_A.copy()

        # Forward-Backward step
        if not pfm_only:
            z_ista_L = y_ista_L + c_ist * data_fidelity

            # Estimate L
            Ut, St, Vt = linalg.svd(
                z_ista_L, full_matrices=False, compute_uv=True, check_finite=True
            )

            if num_iter == 0:
                # Calculate absolute difference between eigenvalues.
                St_diff = abs(np.diff(St) / St[1:])

                # Find what eigenvalue differences are bigger than the threshold.
                keep_diff = np.where(St_diff >= eigen_thr)[0]

                # Use first difference above the threshold as the number of low-rank components.
                keep_idx = keep_diff[0]

                LGR.info(f"{keep_idx+1} low-rank components found")

            St[keep_idx + 1 :] = 0

            if num_iter == 0:
                L = np.dot(np.dot(Ut, np.diag(St) / c_ist), Vt)
                data_fidelity = y - L
            else:
                L = np.dot(np.dot(Ut, np.diag(St)), Vt)
                if block_model:
                    data_fidelity = y - L - S_fitts
                else:
                    data_fidelity = A - L

        if pfm_only:
            S_fidelity = v - np.dot(hrf_cov, y_ista_S)
        else:
            if n_te > 1:
                # Multi-echo: use second echo's data_fidelity directly
                # (already has correct shape for S update)
                S_fidelity = data_fidelity[nscans : 2 * nscans, :]
            else:
                # Single-echo: project data_fidelity through hrf.T
                # This handles both full data and subsampled data cases
                if block_model:
                    S_fidelity = np.dot(hrf_trans, data_fidelity)
                else:
                    S_fidelity = np.dot(hrf_trans, data_fidelity) - np.dot(
                        hrf_cov, y_ista_S
                    )

        z_ista_S = y_ista_S + c_ist * S_fidelity

        # Estimate S
        if group > 0:
            S = proximal_operator_mixed_norm(
                z_ista_S, c_ist * lambda_S, rho_val=(1 - group)
            )
        else:
            S = proximal_operator_lasso(z_ista_S, c_ist * lambda_S)

        #  Perform debiasing to have both S and L on the same amplitude scale
        if not pfm_only:
            if block_model:
                S_spike = S
                S_fitts = np.dot(hrf, S)
            else:
                S, S_fitts = debiasing_spike(
                    hrf=hrf, y=y, estimates_matrix=S, n_jobs=jobs
                )
                S_spike = S
        else:
            S_spike = S
            S_fitts = np.dot(hrf, S)

        if not pfm_only and not block_model:
            A = y_ista_A + np.dot(hrf, S_spike - y_ista_S) + (L - y_ista_L)

        t_fista_old = t_fista
        t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old**2)))

        y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista
        y_fista_L = L + (L - L_old) * (t_fista_old - 1) / t_fista
        y_fista_A = A + (A - A_old) * (t_fista_old - 1) / t_fista

        # Convergence
        if num_iter >= min_iter:
            if pfm_only:
                nonzero_idxs_rows, nonzero_idxs_cols = np.where(
                    np.abs(S) > 10 * np.finfo(float).eps
                )
                diff = np.abs(
                    S[nonzero_idxs_rows, nonzero_idxs_cols]
                    - S_old[nonzero_idxs_rows, nonzero_idxs_cols]
                )
                convergence_criteria = np.abs(
                    diff / S_old[nonzero_idxs_rows, nonzero_idxs_cols]
                )

                if np.all(convergence_criteria <= tol):
                    break
            else:
                # Residuals (only computed for non-pfm_only case)
                nv = np.sqrt(np.sum((S_fitts + L - y) ** 2, axis=0) / nscans)
                # MAD-based convergence only applies when lambda is auto-selected
                # (precision is None when explicit lambd is provided)
                if (
                    precision is not None
                    and any(abs(nv - noise_estimate) < precision)
                    and lambda_crit == "mad_update"
                ):
                    break
                elif not pfm_only and linalg.norm(A - A_old) < tol * linalg.norm(A_old):
                    break
                else:
                    diff = (abs(S_old - S) < tol).flatten()
                    if (np.sum(diff) / len(diff)) > 0.5:
                        break

        # Update lambda (only when auto-selected, not user-provided)
        if update_lambda and lambd is None:
            nv = np.sqrt(np.sum((S_fitts + L - y) ** 2, axis=0) / nscans)
            lambda_S = np.nan_to_num(lambda_S * noise_estimate / nv)

    if not pfm_only:
        # Extract low-rank maps and time-series
        Ut, St, Vt = linalg.svd(
            np.nan_to_num(L), full_matrices=False, compute_uv=True, check_finite=True
        )

        # Make sure the correct number of components are saved.
        eig_vecs = Ut[:, : keep_idx + 1]
        eig_maps = Vt[: keep_idx + 1, :]

        # Normalize low-rank maps and time-series
        mean_eig_vecs = np.mean(eig_vecs, axis=0)
        std_eig_vecs = np.std(eig_vecs, axis=0)
        eig_vecs = (eig_vecs - mean_eig_vecs) / (std_eig_vecs)
        mean_eig_maps = np.expand_dims(np.mean(eig_maps, axis=1), axis=1)
        std_eig_maps = np.expand_dims(np.std(eig_maps, axis=1), axis=1)
        eig_maps = (eig_maps - mean_eig_maps) / std_eig_maps
    else:
        eig_vecs = None
        eig_maps = None

    return S, eig_vecs, eig_maps, noise_estimate, lambda_S, np.nan_to_num(L)
