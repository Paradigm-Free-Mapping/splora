import numpy as np
from scipy import linalg
from sklearn.utils.validation import check_array
from scipy.stats import median_absolute_deviation
from pywt import wavedec


def proximal_operator_lasso(y, lambda_val, weights=0):
    x = y*np.maximum(np.zeros(y.shape), 1-(lambda_val/abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return(x)


def proximal_operator_mixed_norm(y, lambda_val, rho_val=0.8, groups='space'):
    # Division parameter of proximal operator
    div = y / np.abs(y)
    div[np.isnan(div)] = 0

    # First parameter of proximal operator
    p_one = np.maximum(np.zeros(y.shape), (np.abs(y) - lambda_val * rho_val))

    # Second parameter of proximal operator
    if groups == 'space':
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - lambda_val * rho_val) ** 2, axis=1)
        foo = foo.reshape(len(foo), 1)
        foo = np.dot(foo, np.ones((1, y.shape[1])))
    else:
        foo = np.sum(np.maximum(np.zeros(y.shape), np.abs(y) - lambda_val * rho_val) ** 2, axis=0)
        foo = foo.reshape(1, len(foo))
        foo = np.dot(np.ones((y.shape[0], 1), foo))

    p_two = np.maximum(np.zeros(y.shape),
                       np.ones(y.shape) - lambda_val * (1 - rho_val) / np.sqrt(foo))

    # Proximal operation
    x = div * p_one * p_two

    # Return result
    return(x)


def debiasing(x, y, beta):

    beta_out = np.zeros(beta.shape)
    index_voxels = np.unique(np.where(abs(beta) > 10 * np.finfo(float).eps)[1])

    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > 10 * np.finfo(float).eps)[0]

        X_events = x[:, index_events_opt]
        beta2save = np.zeros((beta.shape[0], 1))

        coef_LSfitdebias, residuals, rank, s = linalg.lstsq(
            X_events, y[:, index_voxels[voxidx]], cond=None)
        beta2save[index_events_opt, 0] = coef_LSfitdebias

        beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))

    return beta_out


def fista_update(X, y, max_iter, update_lambda=True, precision=None, eigen_thr=0.25):

    nscans = y.shape[0]
    nvoxels = y.shape[1]

    c_ist = 1 / (linalg.norm(X) ** 2)

    y_fista_S = np.zeros((nscans, nvoxels), dtype=np.float32)
    y_fista_L = y_fista_S.copy()
    y_fista_A = y_fista_S.copy()
    S = y_fista_S.copy()
    L = y_fista_S.copy()
    A = y_fista_S.copy()
    data_fidelity = y_fista_S.copy()

    nv = np.zeros((max_iter, nvoxels))

    keep_idx = 2
    t_fista = 1
    num_iter = 0

    _, cD1 = wavedec(y, 'db3', level=1, axis=0)
    noise_estimate = median_absolute_deviation(cD1) / 0.8095
    lambda_S = noise_estimate

    if precision is None:
        precision = noise_estimate / 100000

    for num_iter in range(max_iter):

        S_old = S.copy()
        L_old = L.copy()
        A_old = A.copy()
        y_ista_S = y_fista_S.copy()
        y_ista_L = y_fista_L.copy()
        y_ista_A = y_fista_A.copy()

        # Forward-Backward step
        z_ista_S = y_ista_S + c_ist * data_fidelity
        z_ista_L = y_ista_L + c_ist * data_fidelity

        print(lambda_S)

        S = proximal_operator_lasso(z_ista_S, c_ist * lambda_S)

        Ut, St, Vt = linalg.svd(z_ista_L, full_matrices=False,
                                compute_uv=True, check_finite=True)
        # if num_iter == 0:
        #     St_diff = abs(np.diff(St) / St[1:])
        #     keep_diff = np.where(St_diff >= eigen_thr)[0]

        #     diff_old = -1
        #     for i in range(len(keep_diff)):
        #         if (keep_diff[i] - diff_old) == 1:
        #             keep_idx = keep_diff[i] + 1
        #         else:
        #             break
        #         diff_old = keep_diff[i]

        St[keep_idx + 1:] = 0
        # lambda_L = St[keep_idx] * 1.01
        L = np.dot(np.dot(Ut, np.diag(St)), Vt)

        A = y_ista_A + np.dot(X, S - y_ista_S) + (L - y_ista_L)

        t_fista_old = t_fista
        t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old ** 2)))

        # beta_deb = debiasing(X, y, S)

        y_fista_S = S + (S - S_old) * (t_fista_old - 1) / t_fista
        y_fista_L = L + (L - L_old) * (t_fista_old - 1) / t_fista
        y_fista_A = A + (A - A_old) * (t_fista_old - 1) / t_fista

        data_fidelity = y - y_fista_A

        nv[num_iter, :] = np.sqrt(np.sum((np.dot(X, S) + L - y) ** 2, axis=0) / nscans)

        # breakpoint()
        if any(abs(nv[num_iter] - noise_estimate) < precision):
            break
        if update_lambda:
            lambda_S = lambda_S * noise_estimate / nv[num_iter]

        num_iter += 1

    # Save estimated bold signals for current iteration
    betafitts = np.dot(X, S.astype(np.float32))

    return S, betafitts, L
