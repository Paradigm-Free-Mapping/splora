import time

import numpy as np
from pywt import wavedec
from scipy.linalg import norm, svd


# soft-thresholding function
def SoftThresh(x, p, is_low_rank=False):
    xa = abs(x)
    xap = xa > p
    y = (xa - p) * (x / xa) * xap
    y[np.invert(xap)] = 0
    return y


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


def low_rank(
    data,
    hrf,
    nt,
    n_te=1,
    maxiter=100,
    miniter=10,
    vox_2_keep=0.3,
    nruns=1,
    lambda_weight=1.5,
    group=0,
    eigen_thr=0.25,
    is_pfm=False,
    update_lambda=True,
):
    """
    L+S reconstruction of undersampled dynamic MRI data using iterative
    soft-thresholding of singular values of L and soft-thresholding of
    sparse representation WS

    Input variables and reconstruction parameters must be stored in the
    struct param

    data: undersampled k-t data (nx,ny,nt,nc)
    param.E: data acquisition operator
    param.W: sparsifying operator
    param.lambda_L: nuclear-norm weight
    param.lambda_S: sparse weight
    param.c: inverse of proximal-gradient step (1/c)  (1/L_k in the paper)
    param.mu: step for extra point \bar{x}=x+mu(z-x)
    param.nite: number of iterations
    param.errortol: stoping tolerance
    param.backtracking: use backtracking (auto-increase L_k)
    param.backstep: step increase rate when using backtracking
    param.ccfista: use same convergence consdition than FISTA in backtracking

    Edited by Marcelo V. W. Zibetti (2018)
    Reference:
    M.V.W. Zibetti, E.S. Helou, R.R. Regatte, and G.T. Herman, "Monotone
    FISTA with Variable Acceleration for Compressed Sensing Magnetic
    Resonance Imaging" IEEE Transactions on Computational Imaging,
    v. ,pp , 2018
    """

    print("MFISTA-AS for L2-L+S problems")

    nvox = data.shape[1]

    _, cD1 = wavedec(data, "db3", level=1, axis=0)
    noise_est = np.median(abs(cD1 - np.median(cD1, axis=0)), axis=0) / 0.6745

    if n_te == 1:
        L = np.zeros((nt, nvox))
    else:
        L = np.zeros((n_te * nt, nvox))
    S = np.zeros((nt, nvox))

    # algorithm parameters
    cc = norm(hrf) ** 2
    mu_in = 1.5
    tol = 1e-6
    restart = False
    comp_cost = False
    display = True

    hrf_trans = hrf.T
    hrf_cov = np.dot(hrf_trans, hrf)
    v = np.dot(hrf_trans, data)

    # iterations
    A = 0
    keep_idx = 1

    # data[abs(data) < 1e-3] = 0

    A = np.dot(hrf, S) + L

    YL = L.copy()
    YS = S.copy()
    YA = A.copy()

    # Initializes cost arrays
    L2cost = np.zeros((maxiter,))
    L1cost = np.zeros((maxiter,))
    Ncost = np.zeros((maxiter,))
    COST = np.zeros((maxiter,))
    ERR = np.zeros((maxiter,))
    t = np.zeros((maxiter,))
    # zeta = np.zeros((maxiter,))
    # eta = np.zeros((maxiter,))
    # delta = np.zeros((maxiter,))
    mu = np.zeros((maxiter,))

    Ut, St, Vt = svd(data, full_matrices=False, compute_uv=True, check_finite=True)

    St_diff = abs(np.diff(St) / St[1:])
    keep_diff = np.where(St_diff >= eigen_thr)[0]

    keep_idx = 1
    diff_old = -1
    for i in range(len(keep_diff)):
        if (keep_diff[i] - diff_old) == 1:
            keep_idx = keep_diff[i] + 1
        else:
            break
        diff_old = keep_diff[i]

    lambda_S = noise_est * lambda_weight

    print(f"Keeping {keep_idx} eigenvalues...")

    lambda_L = St[keep_idx] * 1.01

    if is_pfm:
        lambda_L = 0

    St[keep_idx + 1 :] = 0

    # Residue
    L2cost[i] = 1 / 2 * np.linalg.norm(data.flatten() - A.flatten(), ord=2) ** 2
    L1cost[i] = np.linalg.norm(S.flatten(), ord=1)
    Ls = svd(L, full_matrices=False, compute_uv=False)
    Ncost[i] = np.sum(Ls)
    COST[i] = L2cost[i] + lambda_L * Ncost[i] + np.mean(lambda_S) * L1cost[i]
    # Estimation error
    ERR[i] = np.linalg.norm(data.flatten() - A.flatten(), ord=2)

    t[i] = 1

    for i in range(maxiter - 1):
        # data consistency gradient
        y_YA = data - YA

        LO = L.copy()
        SO = S.copy()
        AO = A.copy()

        # Low-rank update
        if lambda_L != 0:
            Ut, St, Vt = svd(
                np.nan_to_num(YL + (1 / cc) * y_YA),
                full_matrices=False,
                compute_uv=True,
                check_finite=True,
            )
            # St = np.diag(SoftThresh(St, lambda_L / cc, is_low_rank=True))
            St[keep_idx + 1 :] = 0
            St = np.diag(St)
            LZ = np.dot(np.dot(Ut, St), Vt)
        else:
            LZ = np.zeros((L.shape))
            YL = np.zeros((L.shape))

        # Sparse update
        YS_residual = v - np.dot(hrf_cov, YS)
        YSS = YS + (1 / cc) * YS_residual  # y_YA

        if group == 0:
            SZ = SoftThresh(YSS, lambda_S / cc)
        else:
            SZ = proximal_operator_mixed_norm(YSS, lambda_S / cc, rho_val=(1 - group))

        # SZ[abs(SZ) < 5e-4] = 0

        SZ_YS = SZ - YS
        LZ_YL = LZ - YL
        AZ_YA = np.dot(hrf, SZ_YS) + LZ_YL
        AZ = YA + AZ_YA

        dA = AZ - AO
        dS = SZ - SO
        dL = LZ - LO

        # Majorizer gap
        y_AZ = data - AZ
        f_Z = 0.5 * (np.dot(y_AZ.flatten().T, y_AZ.flatten()))
        # f_Y = 0.5 * (np.dot(y_YA.flatten().T, y_YA.flatten()))
        # QdZY = (cc / 2 * np.linalg.norm(LZ_YL.flatten(), ord=2) ** 2) + (
        # cc / 2 * np.linalg.norm(SZ_YS.flatten(), ord=2) ** 2
        # )
        # zeta[i] = (
        #    f_Y
        #    - np.real(
        #        np.dot(YS_residual.flatten().T, SZ_YS.flatten())
        #        + np.dot(y_YA.flatten().T, LZ_YL.flatten())
        #    )
        #    + QdZY
        #    - f_Z
        # )

        LZs = svd(LZ, full_matrices=False, compute_uv=False, check_finite=True)
        COSTCZ = (
            f_Z
            + lambda_L * sum(LZs)
            + np.mean(lambda_S) * np.linalg.norm(SZ.flatten(), ord=1)
        )

        if COSTCZ < COST[i]:
            S = SZ
            L = LZ
            A = AZ
            COSTC = COSTCZ
            mu[i] = 1
        else:
            S = SO
            L = LO
            A = AO
            COSTC = COST[i]
            mu[i] = 0

        if mu_in != 1:
            SS = SO + mu_in * dS
            LS = LO + mu_in * dL
            AS = AO + mu_in * dA

            y_AS = data - AS
            LSs = svd(LS, full_matrices=False, compute_uv=False, check_finite=True)
            COSTCS = (
                np.dot(y_AS.flatten().T, y_AS.flatten()) / 2
                + lambda_L * np.sum(LSs)
                + np.mean(lambda_S) * np.linalg.norm(SS.flatten(), ord=1)
            )

            if COSTCS < COSTCZ:
                if COSTCS < COST[i]:
                    S = SS
                    L = LS
                    A = AS
                    COSTC = COSTCS
                    mu[i] = mu_in
        # S_nonzero = np.count_nonzero(S, axis=1)
        # global_fluc = np.where(S_nonzero > nvox * vox_2_keep)[0]
        # S[global_fluc, :] = 0

        # Alpha step gap
        # delta[i] = -COSTC + COSTCZ

        # Overstep
        # eta[i] = 1 + (zeta[i] + delta[i]) / (QdZY + np.finfo(float).eps)

        S_SO = S - SO
        L_LO = L - LO
        A_AO = A - AO
        SZ_S = SZ - S
        LZ_L = LZ - L
        AZ_A = AZ - A

        # Restart is experimental
        rest1 = mu[i] == 0
        if (rest1) and restart:
            t[i] = 1
            SZ_S = 0
            LZ_L = 0
            AZ_A = 0

        t[i + 1] = (1 + np.sqrt(1 + 4 * t[i] ** 2)) / 2  # Combination parameter

        t1 = (t[i] - 1) / t[i + 1]
        t2 = t[i] / t[i + 1]
        # t3 = (t[i] / t[i + 1]) * (eta[i] - 1)

        YS = S + t1 * (S_SO) + t2 * (SZ_S)  # + t3 * (SZ_YS)
        YL = L + t1 * (L_LO) + t2 * (LZ_L)  # + t3 * (LZ_YL)
        YA = A + t1 * (A_AO) + t2 * (AZ_A)  # + t3 * (AZ_YA)

        y_A = data - A

        if update_lambda:
            nv = np.sqrt(np.sum((np.dot(hrf, S) - data) ** 2, axis=0) / nt)

            if all(abs(nv - noise_est) > (tol / 1e5)):
                lambda_old = lambda_S
                lambda_S = lambda_S * noise_est / nv

        if comp_cost:
            L2cost[i] = np.dot(y_A.flatten().T, y_A.flatten()) / 2  # Residue
            L1cost[i] = np.linalg.norm(S.flatten(), ord=1)
            Ls = svd(L, full_matrices=False, compute_uv=False, check_finite=True)
            Ncost[i] = np.sum(Ls)
            COST[i] = L2cost[i] + lambda_L * Ncost[i] + np.mean(lambda_S) * L1cost[i]
            ERR[i] = np.linalg.norm(data.flatten() - A.flatten(), ord=2) / nt
            # Print some numbers
            if display:
                print(
                    f"mfista-va i={i}, cost={COST[i]:.9f},"
                    f"err={ERR[i]:.9f}, L={cc:.3f}, mu={mu[i-1]:.3f} \n"
                )

        else:
            COST[i] = COSTC
            if display:
                print(f"mfista-va i={i}, L={cc:.3f}, mu={mu[i-1]:.3f} \n")

        # Force at least 10 itereations with no improvement
        if i > (miniter - 1) and (ERR[i] - ERR[i - 1]) < tol:
            break

    # END WHILE

    MSE_iter = np.min(
        np.sqrt(np.sum(abs(((np.dot(hrf, S) + L) - data)) ** 2, axis=0)) / nt
    )

    print(f"MSE is {MSE_iter}")

    # S_nonzero = np.count_nonzero(S, axis=1)
    # global_fluc = np.where(S_nonzero > nvox * vox_2_keep)[0]
    # S[global_fluc, :] = 0

    # Return eigen vectors we keep.
    Ut, St, Vt = svd(
        np.nan_to_num(L), full_matrices=False, compute_uv=True, check_finite=True
    )

    eig_vecs = Ut[:, :keep_idx]
    mean_eig_vecs = np.mean(eig_vecs, axis=0)
    std_eig_vecs = np.std(eig_vecs, axis=0)
    eig_vecs = (eig_vecs - mean_eig_vecs) / (std_eig_vecs)
    eig_maps = Vt[:keep_idx, :]
    mean_eig_maps = np.expand_dims(np.mean(eig_maps, axis=1), axis=1)
    std_eig_maps = np.expand_dims(np.std(eig_maps, axis=1), axis=1)
    eig_maps = (eig_maps - mean_eig_maps) / std_eig_maps

    return (L, S, eig_vecs, eig_maps)
