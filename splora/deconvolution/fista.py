"""FISTA solver for PFM."""
import logging
from os.path import join as opj

import numpy as np
from pywt import wavedec
from scipy import linalg
from scipy.stats import median_absolute_deviation
from splora.io import write_data

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
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=1)
        foo = foo.reshape(len(foo), 1)
        foo = np.dot(foo, np.ones((1, y.shape[1])))
    else:
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - thr * rho_val) ** 2, axis=0)
        foo = foo.reshape(1, len(foo))
        foo = np.dot(np.ones((y.shape[0], 1), foo))

    p_two = np.maximum(
        np.zeros(y.shape), np.ones(y.shape) - np.nan_to_num(thr * (1 - rho_val) / np.sqrt(foo))
    )

    # Proximal operation
    x = div * p_one * p_two

    # Return result
    return x


def select_lambda(hrf, y, criteria="mad_update", factor=1, pcg=0.7):
    """Criteria to select regularization parameter lambda.

    Parameters
    ----------
    hrf : (E x T) array_like
        Matrix containing shifted HRFs in its columns. E stands for the number of volumes times
        the number of echo-times.
    y : (T x S) array_like
        Matrix with fMRI data provided to splora.
    criteria : str, optional
        Criteria to select regularization parameter lambda, by default "mad_update"
    factor : int, optional
        Factor by which to multiply the value of lambda, by default 1
        Only used when "factor" criteria is selected.
    pcg : str, optional
        Percentage of maximum lambda possible to use, by default "0.7"
        Only used when "pcg" criteria is selected.

    Returns
    -------
    lambda_selection : array_like
        Value of the regularization parameter lambda for each voxel.
    update_lambda : bool
        Whether to update lambda after each iteration until it converges to the MAD estimate
        of the noise.
    noise_estimate : array_like
        MAD estimate of the noise.
    """
    update_lambda = False
    nt = hrf.shape[1]

    # Use last echo to estimate noise
    if hrf.shape[0] > nt:
        y = y[-nt:, :]

    _, cD1 = wavedec(y, "db3", level=1, axis=0)
    noise_estimate = median_absolute_deviation(cD1) / 0.6745  # 0.8095

    if criteria == "mad":
        lambda_selection = noise_estimate
    elif criteria == "mad_update":
        lambda_selection = noise_estimate
        update_lambda = True
    elif criteria == "ut":
        lambda_selection = noise_estimate * np.sqrt(2 * np.log10(nt))
    elif criteria == "lut":
        lambda_selection = noise_estimate * np.sqrt(
            2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt))
        )
    elif criteria == "factor":
        lambda_selection = noise_estimate * factor
    elif criteria == "pcg":
        max_lambda = np.mean(abs(np.dot(hrf.T, y)), axis=0)
        lambda_selection = max_lambda * pcg
    elif criteria == "eigval":
        random_signal = np.random.normal(loc=0.0, scale=np.mean(noise_estimate), size=y.shape)
        s = np.linalg.svd(random_signal, compute_uv=False)
        lambda_selection = s[0]

    return lambda_selection, update_lambda, noise_estimate


def fista(
    hrf,
    y,
    n_te,
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
    nvoxels = y.shape[1]
    nscans = hrf.shape[1]

    c_ist = 1 / (linalg.norm(hrf) ** 2)
    hrf_trans = hrf.T
    hrf_cov = np.dot(hrf_trans, hrf)
    v = np.dot(hrf_trans, y)

    y_fista_S = np.zeros((nscans, nvoxels), dtype=np.float32)
    y_fista_L = np.zeros(y.shape)
    fitt = y_fista_L.copy()
    y_fista_A = y_fista_L.copy()
    S = y_fista_S.copy()
    if n_te == 1:
        L = np.zeros((nscans, nvoxels))
    else:
        L = np.zeros((n_te * nscans, nvoxels))
    A = L.copy()
    data_fidelity = y - y_fista_L

    keep_idx = 1
    t_fista = 1

    # Select lambda for each voxel based on criteria
    lambda_S, update_lambda, noise_estimate = select_lambda(
        hrf, y, factor=factor, criteria=lambda_crit
    )

    # LGR.info(f"Selected lambda is {lambda_S}")

    if precision is None:
        precision = noise_estimate / 100000

    # Perform FISTA
    for num_iter in range(max_iter):

        LGR.info(f"Iteration {num_iter + 1}/{max_iter}")

        data_fidelity = y - y_fista_A

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
                St_diff = abs(np.diff(St) / St[1:])
                keep_diff = np.where(St_diff >= eigen_thr)[0]

                diff_old = -1
                for i in range(len(keep_diff)):
                    if (keep_diff[i] - diff_old) == 1:
                        keep_idx = keep_diff[i] + 1
                    else:
                        break
                    diff_old = keep_diff[i]

                LGR.info(f"{keep_idx+1} low-rank components found")

            St[keep_idx + 1 :] = 0
            L = np.dot(np.dot(Ut, np.diag(St)), Vt)

        S_fidelity = data_fidelity[:nscans, :]
        z_ista_S = y_ista_S + c_ist * S_fidelity

        # Estimate S
        if group > 0:
            S = proximal_operator_mixed_norm(z_ista_S, c_ist * lambda_S, rho_val=(1 - group))
        else:
            S = proximal_operator_lasso(z_ista_S, c_ist * lambda_S)

        A = y_ista_A + np.dot(hrf, S - y_ista_S) + (L - y_ista_L)

        t_fista_old = t_fista
        t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old ** 2)))

        y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista
        y_fista_L = L + (L - L_old) * (t_fista_old - 1) / t_fista
        y_fista_A = A + (A - A_old) * (t_fista_old - 1) / t_fista

        # Residuals
        nv = np.sqrt(np.sum((fitt + L - y) ** 2, axis=0) / nscans)

        # Convergence
        if num_iter >= min_iter:
            if any(abs(nv - noise_estimate) < precision) and lambda_crit == "mad_update":
                break
            elif linalg.norm(A - A_old) < tol * linalg.norm(A_old):
                break
            else:
                diff = (abs(S_old - S) < tol).flatten()
                if (np.sum(diff) / len(diff)) > 0.5:
                    break

        # Update lambda
        if update_lambda:
            lambda_S = lambda_S * noise_estimate / nv

    if not pfm_only:
        # Extract low-rank maps and time-series
        Ut, St, Vt = linalg.svd(
            np.nan_to_num(L), full_matrices=False, compute_uv=True, check_finite=True
        )

        # Normalize low-rank maps and time-series
        eig_vecs = Ut[:, :keep_idx]
        mean_eig_vecs = np.mean(eig_vecs, axis=0)
        std_eig_vecs = np.std(eig_vecs, axis=0)
        eig_vecs = (eig_vecs - mean_eig_vecs) / (std_eig_vecs)
        eig_maps = Vt[:keep_idx, :]
        mean_eig_maps = np.expand_dims(np.mean(eig_maps, axis=1), axis=1)
        std_eig_maps = np.expand_dims(np.std(eig_maps, axis=1), axis=1)
        eig_maps = (eig_maps - mean_eig_maps) / std_eig_maps
    else:
        eig_vecs = None
        eig_maps = None

    return S, eig_vecs, eig_maps, noise_estimate, lambda_S
