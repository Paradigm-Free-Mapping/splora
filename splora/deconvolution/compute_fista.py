from splora.deconvolution.fista import fista
import numpy as np
import os

# Get variables from environment variables
lambda_values = os.getenv("LAMBDAS")
lambda_values = np.load(lambda_values)
data = os.getenv("DATA")
data = np.load(data)
hrf = os.getenv("HRF")
hrf = np.load(hrf)
nTE = os.getenv("nTE")
nTE = int(nTE)
group = os.getenv("GROUP")
group = float(group)
block = os.getenv("BLOCK")
block = block == "True"
tr = os.getenv("TR")
tr = float(tr)
jobs = os.getenv("JOBS")
jobs = int(jobs)
n_sur = os.getenv("NSURR")
n_sur = int(n_sur)
temp = os.getenv("TEMP")
nscans = os.getenv("NSCANS")
nscans = int(nscans)

print(f"Lambdas: {lambda_values}")
print(f"Data: {data}")
print(f"HRF: {hrf}")
print(f"nTE: {nTE}")
print(f"group: {group}")
print(f"block: {block}")
print(f"tr: {tr}")
print(f"jobs: {jobs}")
print(f"n_sur: {n_sur}")
print(f"temp: {temp}")
print(f"nscans: {nscans}")

def subsample(nscans, mode, nTE):
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


# Subsample the data
subsample_idx = subsample(nscans, 1, nTE)
data_sub = data[subsample_idx, :]
hrf_sub = hrf[subsample_idx, :]

# Number of lambdas
n_lambdas = lambda_values.shape[0]

# Iterate through all the lambda values
for lambda_idx in range(n_lambdas):

    S = fista(
        hrf=hrf_sub,
        y=data_sub,
        n_te=nTE,
        group=group,
        pfm_only=True,
        block_model=block,
        tr=tr,
        jobs=jobs,
        lambd=lambda_values[lambda_idx, :],
    )[0]

    # Output filename
    filename = f"beta_{n_sur}_{lambda_idx}.npy"
    print(f"Finished FISTA for lambda {lambda_idx}")

    # Save boolean of beta values to npy file
    np.save(os.path.join(temp, filename), S != 0)
