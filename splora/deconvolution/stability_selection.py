"""Stability selection for the deconvolution problem."""

import logging

import numpy as np
from dask import compute
from dask import delayed as delayed_dask

from splora.deconvolution.fista import fista

LGR = logging.getLogger("GENERAL")


def get_subsampling_indices(n_scans, n_te=1, ratio=0.6):
    """Get subsampling indices for stability selection.

    Parameters
    ----------
    n_scans : int
        The number of scans.
    n_te : int, optional
        The number of echo times, by default 1.
    ratio : float, optional
        The ratio of time points to keep, by default 0.6.

    Returns
    -------
    subsample_idx : np.ndarray
        The indices of the subsampled data.
    """
    n_keep = int(ratio * n_scans)
    subsample_idx = np.sort(np.random.choice(range(n_scans), n_keep, replace=False))

    if n_te > 1:
        # Same time points across echoes
        all_indices = subsample_idx.copy()
        for i in range(1, n_te):
            all_indices = np.concatenate((all_indices, subsample_idx + i * n_scans))
        subsample_idx = all_indices

    return subsample_idx


def calculate_lambda_range(hrf, y, n_lambdas=30):
    """Calculate the lambda range for each voxel.

    Parameters
    ----------
    hrf : np.ndarray
        The hemodynamic response function matrix.
    y : np.ndarray
        The data matrix (n_timepoints x n_voxels).
    n_lambdas : int, optional
        The number of lambda values, by default 30.

    Returns
    -------
    lambda_values : np.ndarray
        The lambda values (n_lambdas x n_voxels).
    """
    n_voxels = y.shape[1]
    lambda_values = np.zeros((n_lambdas, n_voxels))

    for voxel in range(n_voxels):
        voxel_data = y[:, voxel]
        max_lambda = abs(np.dot(hrf.T, voxel_data)).max()

        if max_lambda > 0:
            lambda_values[:, voxel] = np.geomspace(
                0.05 * max_lambda, 0.95 * max_lambda, n_lambdas
            )
        else:
            lambda_values[:, voxel] = np.inf * np.ones(n_lambdas)

    return lambda_values


def run_surrogate(
    hrf,
    y,
    n_scans,
    n_te,
    n_lambdas,
    group,
    block_model,
    tr,
    te,
    max_iter,
    min_iter,
    seed,
    pfm_only=False,
    jobs=4,
):
    """Run FISTA for all lambda values for a single surrogate.

    This function runs FISTA on subsampled data for all lambda values.
    Each surrogate uses different random subsampling of the data.

    For both pfm_only=True (sparse only) and pfm_only=False (low-rank + sparse),
    the data is subsampled and full FISTA is run with iterative refinement.

    Parameters
    ----------
    hrf : np.ndarray
        The HRF matrix (n_samples x n_scans).
    y : np.ndarray
        The data matrix (n_samples x n_voxels).
    n_scans : int
        The number of scans (columns of HRF / output timepoints).
    n_te : int
        The number of echo times.
    n_lambdas : int
        The number of lambda values.
    group : float
        The group sparsity weight.
    block_model : bool
        Whether to use a block model.
    tr : float
        The repetition time.
    te : list
        The echo times.
    max_iter : int
        Maximum number of FISTA iterations.
    min_iter : int
        Minimum number of FISTA iterations.
    seed : int
        Random seed for subsampling reproducibility.
    pfm_only : bool, optional
        Whether to run without low-rank estimation, by default False.
    jobs : int, optional
        Number of jobs for debiasing step, by default 4.

    Returns
    -------
    results : np.ndarray
        Boolean array of non-zero coefficients (n_lambdas x n_scans x n_voxels).
    lambda_values : np.ndarray
        Lambda values used (n_lambdas x n_voxels).
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Get subsampling indices for rows (observations)
    subsample_idx = get_subsampling_indices(n_scans, n_te)

    # Subsample rows of HRF and y (observations)
    hrf_sub = hrf[subsample_idx, :]
    y_sub = y[subsample_idx, :]

    # Calculate lambda range for subsampled data
    lambda_values = calculate_lambda_range(hrf_sub, y_sub, n_lambdas)

    n_voxels = y.shape[1]

    # Results array - S output will still be (n_scans x n_voxels)
    results = np.zeros((n_lambdas, n_scans, n_voxels), dtype=np.int8)

    # Run FISTA for each lambda value
    # For low-rank + sparse (pfm_only=False), FISTA alternates between L and S
    # with proper gradient projection to handle subsampled data
    for lambda_idx in range(n_lambdas):
        S, _, _, _, _, _ = fista(
            hrf=hrf_sub,
            y=y_sub,
            n_te=n_te,
            lambd=lambda_values[lambda_idx, :],
            max_iter=max_iter,
            min_iter=min_iter,
            group=group,
            pfm_only=pfm_only,
            block_model=block_model,
            tr=tr,
            te=te,
            jobs=jobs,
        )

        # S has shape (n_scans, n_voxels) - full time series
        nonzero = (np.abs(S) > np.finfo(float).eps).astype(np.int8)
        results[lambda_idx, :, :] = nonzero

    return results, lambda_values


def calculate_auc(all_results, n_surrogates):
    """Calculate the Area Under the Curve (AUC) for stability selection.

    Parameters
    ----------
    all_results : list
        List of (results, lambda_values) tuples from each surrogate.
        results shape: (n_lambdas, n_scans, n_voxels)
        lambda_values shape: (n_lambdas, n_voxels)
    n_surrogates : int
        The number of surrogates.

    Returns
    -------
    auc : np.ndarray
        The AUC values (n_scans x n_voxels).
    """
    # Get dimensions from first result
    first_results, first_lambdas = all_results[0]
    n_lambdas, n_scans, n_voxels = first_results.shape

    # Accumulate selection frequencies and lambda values
    selection_sum = np.zeros((n_lambdas, n_scans, n_voxels))
    lambda_sum = np.zeros((n_lambdas, n_voxels))

    for results, lambda_values in all_results:
        selection_sum += results
        lambda_sum += lambda_values

    # Average selection frequencies
    selection_freq = selection_sum / n_surrogates

    # Average lambda values and normalize
    lambda_avg = lambda_sum / n_surrogates
    lambda_total = np.sum(lambda_avg, axis=0)
    # Avoid division by zero
    lambda_total = np.where(lambda_total == 0, 1, lambda_total)
    lambda_weights = lambda_avg / lambda_total[np.newaxis, :]

    # Calculate AUC as weighted sum
    auc = np.zeros((n_scans, n_voxels))
    for lambda_idx in range(n_lambdas):
        auc += selection_freq[lambda_idx, :, :] * lambda_weights[lambda_idx, :]

    return auc


def stability_selection(
    hrf,
    y,
    n_te,
    tr,
    n_scans,
    block_model=False,
    n_jobs=4,
    n_lambdas=30,
    n_surrogates=30,
    group=0.2,
    te=None,
    max_iter=100,
    min_iter=10,
    pfm_only=False,
):
    """Perform stability selection using dask parallelization.

    Stability selection runs FISTA on multiple subsampled versions of the data
    (surrogates) with a range of lambda values. The selection frequency of each
    timepoint across surrogates and lambdas is used to compute an AUC score.

    For both low-rank + sparse mode (pfm_only=False) and sparse-only mode
    (pfm_only=True), subsampling is applied to the input data before FISTA runs.
    FISTA is run directly on the subsampled data for each surrogate, regardless
    of the pfm_only parameter.

    The surrogates are parallelized using dask, while each surrogate runs
    FISTA sequentially for all lambda values (since FISTA needs all voxels
    at once for the low-rank estimation).

    Parameters
    ----------
    hrf : np.ndarray
        The hemodynamic response function matrix.
    y : np.ndarray
        The data matrix (n_timepoints x n_voxels).
    n_te : int
        The number of echo times.
    tr : float
        The repetition time.
    n_scans : int
        The number of scans.
    block_model : bool, optional
        Whether to use a block model, by default False.
    n_jobs : int, optional
        The number of parallel jobs, by default 4.
    n_lambdas : int, optional
        The number of lambda values to use, by default 30.
    n_surrogates : int, optional
        The number of surrogates to use, by default 30.
    group : float, optional
        The group sparsity weight, by default 0.2.
    te : list, optional
        List of echo times, by default None.
    max_iter : int, optional
        Maximum number of FISTA iterations, by default 100.
    min_iter : int, optional
        Minimum number of FISTA iterations, by default 10.
    pfm_only : bool, optional
        Whether to run without low-rank estimation, by default False.

    Returns
    -------
    auc : np.ndarray
        The AUC values (n_scans x n_voxels).
    """
    if te is None:
        te = [1]

    LGR.info(
        f"Starting stability selection with {n_surrogates} surrogates, "
        f"{n_lambdas} lambda values, and {n_jobs} parallel jobs..."
    )
    if not pfm_only:
        LGR.info("Using low-rank + sparse model with iterative L/S refinement.")

    # Create delayed tasks for each surrogate
    # Each surrogate runs full FISTA (with L+S alternation if pfm_only=False)
    # on subsampled data
    futures = [
        delayed_dask(run_surrogate, pure=False)(
            hrf,
            y,
            n_scans,
            n_te,
            n_lambdas,
            group,
            block_model,
            tr,
            te,
            max_iter,
            min_iter,
            seed=surrogate_idx,
            pfm_only=pfm_only,
            jobs=n_jobs,
        )
        for surrogate_idx in range(n_surrogates)
    ]

    # Execute all surrogates in parallel
    LGR.info(f"Running {n_surrogates} surrogates in parallel...")
    if n_jobs > 1:
        all_results = compute(*futures, scheduler="threads", num_workers=n_jobs)
    else:
        all_results = compute(*futures, scheduler="single-threaded")

    LGR.info("All surrogates completed. Computing AUC...")

    # Calculate AUC from all results
    auc = calculate_auc(all_results, n_surrogates)

    LGR.info("Stability selection completed.")

    return auc
