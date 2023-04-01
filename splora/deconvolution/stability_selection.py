"""Stability selection for the deconvolution problem."""
import logging
from os.path import join as opj

import numpy as np
from dask import compute
from dask import delayed as delayed_dask
from pySPFM import utils

from splora.deconvolution import fista

LGR = logging.getLogger("GENERAL")


def subsample(nscans, mode, nTE):
    """Subsample the data.

    Parameters
    ----------
    nscans : int
        The number of scans.
    mode : int
        The subsampling mode.
    nTE : int
        The number of echo times.

    Returns
    -------
    subsample_idx : array
        The indices of the subsampled data.
    """
    # Subsampling for Stability Selection
    if mode == 1:  # different time points are selected across echoes
        subsample_idx = np.sort(
            np.random.choice(range(nscans), int(0.6 * nscans), 0)
        )  # 60% of timepoints are kept
        if nTE > 1:
            for i in range(nTE - 1):
                subsample_idx = np.concatenate(
                    (
                        subsample_idx,
                        np.sort(
                            np.random.choice(
                                range((i + 1) * nscans, (i + 2) * nscans), int(0.6 * nscans), 0
                            )
                        ),
                    )
                )

    elif mode > 1:  # same time points are selected across echoes
        subsample_idx = np.sort(
            np.random.choice(range(nscans), int(0.6 * nscans), 0)
        )  # 60% of timepoints are kept

    return subsample_idx


def stability_selection(
    hrf,
    y,
    nTE,
    tr,
    temp,
    n_scans,
    block_model=False,
    jobs=4,
    n_lambdas=30,
    n_surrogates=30,
    group=0.2,
    jobqueue=None,
):
    """Perform stability selection on the data.

    Parameters
    ----------
    hrf : array
        The hemodynamic response function.
    y : array
        The data.
    nTE : int
        The number of echo times.
    tr : float
        The repetition time.
    temp : str
        The path to the temporary directory.
    n_scans : int
        The number of scans.
    block_model : bool, optional
        Whether to use a block model, by default False.
    jobs : int, optional
        The number of jobs to run in parallel, by default 4.
    n_lambdas : int, optional
        The number of lambda values to use, by default 30.
    n_surrogates : int, optional
        The number of surrogates to use, by default 30.
    group : float, optional
        The group sparsity, by default 0.2.
    saved_data : bool, optional
        Whether the data has already been saved, by default False.
    jobqueue : str, optional
        The jobqueue to use, by default None.

    Returns
    -------
    auc : array
        The auc values.
    """
    # Get n_scans and n_voxels from y
    n_voxels = y.shape[1]

    # Initialize a matrix of zeros with size n_lambdas n_voxels x n_lambdas
    lambda_values = np.zeros((n_lambdas, n_voxels))

    # For each voxel calculate the lambda values
    for voxel in range(n_voxels):
        # Get the voxel data
        voxel_data = y[:, voxel]

        # Calculate the maximum lambda possible
        max_lambda = abs(np.dot(hrf.T, voxel_data)).max()

        LGR.debug(f"Maximum lambda for voxel {voxel} is {max_lambda}")

        # Calculate the lambda values in a log scale from 0.05 to 0.95 percent
        # of the maximum lambda if the maximum lambda is not zero.
        # Otherwise, make it all np.inf
        if max_lambda > 0:
            lambda_values[:, voxel] = np.geomspace(0.05 * max_lambda, 0.95 * max_lambda, n_lambdas)
        else:
            lambda_values[:, voxel] = np.inf * np.ones(n_lambdas)

    # Save the lambda values to a npy file
    np.save(opj(temp, "lambda_range.npy"), lambda_values)

    # Save hrf and y into npy files
    np.save(opj(temp, "hrf.npy"), hrf)
    np.save(opj(temp, "data.npy"), y)

    # Initiate cluster
    client, _ = utils.dask_scheduler(jobs, jobqueue)

    # Iterate through the number of surrogates and send jobs to the cluster
    # to perform stability selection
    for _ in range(n_surrogates):
        subsample_idx = subsample(n_scans, 1, nTE)

        # Scatter data to workers if client is not None
        if client is not None:
            hrf_fut = client.scatter(hrf[subsample_idx, :])
            y_fut = client.scatter(y[subsample_idx, :])
        else:
            hrf_fut = hrf[subsample_idx, :]
            y_fut = y[subsample_idx, :]

        futures = [
            delayed_dask(fista.fista)(
                hrf=hrf_fut,
                y=y_fut,
                n_te=nTE,
                dims=(n_scans, n_voxels),
                lambd=lambda_values[lambda_, :],
                pfm_only=True,
                group=group,
                block_model=block_model,
                tr=tr,
            )
            for lambda_ in range(n_lambdas)
        ]

        estimates = (
            compute(futures)[0]
            if client is not None
            else compute(futures, scheduler="single-threaded")[0]
        )

        LGR.info(f"Estimates shape: {estimates.shape}")

    # Read the results from the cluster for each surrogate and lambda
    for lambda_idx in range(n_lambdas):
        result = np.squeeze(estimates[lambda_idx])

        print(result.shape)
        for surrogate in range(n_surrogates):
            if surrogate == 0:
                auc_sum = result.astype(int)
            else:
                auc_sum += result.astype(int)

        if lambda_idx == 0:
            auc = (
                auc_sum
                / n_surrogates
                * lambda_values[lambda_idx, :]
                / np.sum(lambda_values, axis=0)
            )
        else:
            auc += (
                auc_sum
                / n_surrogates
                * lambda_values[lambda_idx, :]
                / np.sum(lambda_values, axis=0)
            )

    # Return the auc values
    return auc
