import numpy as np
from pywt import wavedec
from scipy import linalg
from scipy.stats import median_absolute_deviation


def proximal_operator_lasso(y, lambda_val, weights=0):
    x = y * np.maximum(np.zeros(y.shape), 1 - (lambda_val / abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return x


def proximal_operator_mixed_norm(y, lambda_val, rho_val=0.8, groups="space"):
    # Division parameter of proximal operator
    div = y / np.abs(y)
    div[np.isnan(div)] = 0

    # First parameter of proximal operator
    p_one = np.maximum(np.zeros(y.shape), (np.abs(y) - lambda_val * rho_val))

    # Second parameter of proximal operator
    if groups == "space":
        foo = np.sum(
            np.maximum(np.zeros(y.shape), np.abs(y) - lambda_val * rho_val) ** 2, axis=1
        )
        foo = foo.reshape(len(foo), 1)
        foo = np.dot(foo, np.ones((1, y.shape[1])))
    else:
        foo = np.sum(
            np.maximum(np.zeros(y.shape), np.abs(y) - lambda_val * rho_val) ** 2, axis=0
        )
        foo = foo.reshape(1, len(foo))
        foo = np.dot(np.ones((y.shape[0], 1), foo))

    p_two = np.maximum(
        np.zeros(y.shape), np.ones(y.shape) - lambda_val * (1 - rho_val) / np.sqrt(foo)
    )

    # Proximal operation
    x = div * p_one * p_two

    # Return result
    return x


def select_lambda(x, y, criteria="mad_update", pcg="0.7"):
    """
    Select lambda.
    """
    update_lambda = False
    nt = x.shape[1]

    _, cD1 = wavedec(y, "db3", level=1, axis=0)
    noise_estimate = median_absolute_deviation(cD1) / 0.6745  # 0.8095

    if criteria == "mad":
        lambda_selec = noise_estimate
    elif criteria == "mad_update":
        lambda_selec = noise_estimate
        update_lambda = True
    elif criteria == "ut":
        lambda_selec = noise_estimate * np.sqrt(2 * np.log10(nt))
    elif criteria == "lut":
        lambda_selec = noise_estimate * np.sqrt(
            2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt))
        )
    elif criteria == "pcg":
        max_lambda = np.mean(abs(np.dot(x.T, y)), axis=0)
        lambda_selec = max_lambda * pcg

    return lambda_selec, update_lambda, noise_estimate


def fista(
    X,
    y,
    nscans,
    n_te,
    max_iter=100,
    min_iter=10,
    lambda_crit="mad_update",
    precision=None,
    eigen_thr=0.1,
    tol=1e-6,
):

    nvoxels = y.shape[1]

    c_ist = 1 / (linalg.norm(X) ** 2)
    X_trans = X.T
    X_cov = np.dot(X_trans, X)
    v = np.dot(X_trans, y)

    y_fista_S = np.zeros((nscans, nvoxels), dtype=np.float32)
    y_fista_L = np.zeros(y.shape)
    y_fista_A = y_fista_L.copy()
    S = y_fista_S.copy()
    if n_te == 1:
        L = np.zeros((nscans, nvoxels))
    else:
        L = np.zeros((n_te * nscans, nvoxels))
    A = L.copy()
    data_fidelity = L.copy()

    nv = np.zeros((max_iter, nvoxels))

    keep_idx = 1
    t_fista = 1

    # Estimation of the number of low-rank components to keep
    Ut, St, Vt = linalg.svd(y, full_matrices=False, compute_uv=True, check_finite=True)

    St_diff = abs(np.diff(St) / St[1:])
    keep_diff = np.where(St_diff >= eigen_thr)[0]

    diff_old = -1
    for i in range(len(keep_diff)):
        if (keep_diff[i] - diff_old) == 1:
            keep_idx = keep_diff[i] + 1
        else:
            break
        diff_old = keep_diff[i]

    print(f"{keep_idx + 1} low-rank components found.")

    # Select lambda for each voxel based on criteria
    lambda_S, update_lambda, noise_estimate = select_lambda(X, y, criteria=lambda_crit)

    if precision is None:
        precision = noise_estimate / 100000

    # Perform FISTA
    for num_iter in range(max_iter):

        print(f"Iteration {num_iter + 1}/{max_iter}")

        # Save results from previous iteration
        S_old = S.copy()
        L_old = L.copy()
        A_old = A.copy()
        y_ista_S = y_fista_S.copy()
        y_ista_L = y_fista_L.copy()
        y_ista_A = y_fista_A.copy()

        # Forward-Backward step
        S_residuals = v - np.dot(X_cov, y_ista_S)
        z_ista_S = y_ista_S + c_ist * S_residuals
        z_ista_L = y_ista_L + c_ist * data_fidelity

        # Estimate S
        S = proximal_operator_lasso(z_ista_S, c_ist * lambda_S)

        # Estimate L
        Ut, St, Vt = linalg.svd(
            z_ista_L, full_matrices=False, compute_uv=True, check_finite=True
        )
        St[keep_idx + 1 :] = 0
        L = np.dot(np.dot(Ut, np.diag(St)), Vt)

        A = y_ista_A + np.dot(X, S - y_ista_S) + (L - y_ista_L)

        t_fista_old = t_fista
        t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old ** 2)))

        y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista
        y_fista_L = L + (L - L_old) * (t_fista_old - 1) / t_fista
        y_fista_A = A + (A - A_old) * (t_fista_old - 1) / t_fista

        data_fidelity = y - y_fista_A

        # Residuals
        nv[num_iter, :] = np.sqrt(np.sum((np.dot(X, S) + L - y) ** 2, axis=0) / nscans)

        # Convergence
        if num_iter >= min_iter:
            if any(abs(nv[num_iter] - noise_estimate) < precision):
                break
            elif abs(S_old - S) < tol:
                break

        # Update lambda
        if update_lambda:
            lambda_S = lambda_S * noise_estimate / nv[num_iter]

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

    return S, L, eig_vecs, eig_maps
