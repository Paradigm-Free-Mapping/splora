"""Stability selection for the deconvolution problem."""
import logging
import subprocess
import time
from os.path import join as opj

import numpy as np

LGR = logging.getLogger("GENERAL")


def bget(cmd):
    """Run a command on the cluster and return the output.

    Parameters
    ----------
    cmd : str
        The command to run.

    Returns
    -------
    list
        The output of the command.
    """
    from subprocess import PIPE, Popen

    out = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, _) = out.communicate()
    return stdout.decode().split()


def send_job(
    jobname, lambda_values, data, hrf, nTE, group, block_model, tr, jobs, n_sur, temp, nscans
):
    """Send a job to the cluster to perform stability selection.

    Parameters
    ----------
    jobname : str
        The name of the job.
    lambda_values : str
        The path to the lambda values.
    data : str
        The path to the data.
    hrf : str
        The path to the hrf.
    nTE : int
        The number of echo times.
    group : float
        The group sparsity.
    block_model : bool
        Whether to use a block model.
    tr : float
        The repetition time.
    jobs : int
        The number of jobs to run in parallel.
    n_sur : int
        The number of surrogates.
    temp : str
        The path to the temporary directory.
    nscans : int
        The number of scans.
    """
    env_vars = (
        f"LAMBDAS={lambda_values},DATA={data},HRF={hrf},nTE={nTE},GROUP={group},"
        f"BLOCK={block_model},TR={tr},JOBS={jobs},NSURR={n_sur},TEMP={temp},NSCANS={nscans}"
    )
    subprocess.call("module purge", shell=True)
    subprocess.call(
        f"sbatch -J {jobname} -o /scratch/enekouru/{jobname} -e /scratch/enekouru/{jobname} "
        f"--export={env_vars} /scratch/enekouru/splora/splora/deconvolution/compute_fista.slurm",
        shell=True,
    )


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
    saved_data=False,
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

    # Iterate through the number of surrogates and send jobs to the cluster
    # to perform stability selection
    if not saved_data:
        for surrogate in range(n_surrogates):
            jobname = f"stabsel_{surrogate}"

            send_job(
                jobname,
                opj(temp, "lambda_range.npy"),
                opj(temp, "data.npy"),
                opj(temp, "hrf.npy"),
                nTE,
                group,
                block_model,
                tr,
                jobs,
                surrogate,
                temp,
                n_scans,
            )

        while int(bget(f"ls {str(temp)}/beta_* | wc -l")[0]) < (n_surrogates * n_lambdas):
            time.sleep(0.5)

    # Read the results from the cluster for each surrogate and lambda
    for lambda_idx in range(n_lambdas):
        for surrogate in range(n_surrogates):
            # Read the result from the cluster
            result = np.load(opj(temp, f"beta_{surrogate}_{lambda_idx}.npy"))
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
