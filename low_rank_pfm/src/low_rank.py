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
    maxiter=100,
    miniter=10,
    vox_2_keep=0.3,
    nruns=1,
    lambda_weight=1.1,
    group=0,
    eigen_thr=0.25,
    is_pfm=False,
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

    nt = data.shape[0]
    nvox = data.shape[1]

    _, cD1 = wavedec(data, "db3", level=1, axis=0)

    noise_est = np.median(abs(cD1 - np.median(cD1, axis=0)), axis=0) / 0.6745

    nv_2_save = np.zeros((nvox, 50))
    L = np.zeros((nt, nvox))
    S = np.zeros((nt, nvox))

    # algorithm parameters
    cc = norm(hrf) ** 2

    tol = 1e-6

    # iterations
    l_iter = 0
    A = 0
    lambda_S = 0
    l_final = 0

    for ii in range(nruns):
        if l_iter == 0:
            A = np.dot(hrf, S) + L

        YL = L.copy()
        YS = S.copy()
        YA = A.copy()

        t = np.zeros((maxiter,))

        # if l_iter == 0:
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

        lambda_S = noise_est * np.sqrt(
            2 * np.log10(nt) - np.log10(1 + 4 * np.log10(nt))
        )  # * np.sqrt(2)

        lambda_L = np.sqrt(2 * nvox) * (np.median(abs(cD1 - np.median(cD1))) / 0.6745)

        nv = np.ones((nvox,))

        nv_2_save[:, l_iter] = nv
        St[keep_idx + 1 :] = 0

        t[0] = 1

        for i in range(maxiter - 1):
            # data consistency gradient
            y_YA = data - YA

            LO = L.copy()
            SO = S.copy()
            AO = A.copy()

            # Low-rank update
            YLL = YL + (1 / cc) * y_YA

            Ut, St, Vt = svd(
                np.nan_to_num(YLL),
                full_matrices=False,
                compute_uv=True,
                check_finite=True,
            )

            print(St[St > lambda_L / cc])
            St = np.diag(SoftThresh(St, lambda_L / cc, is_low_rank=True))
            LZ = np.dot(np.dot(Ut, St), Vt)

            # Sparse update
            YSS = YS + (1 / cc) * y_YA

            if group == 0:
                SZ = SoftThresh(YSS, lambda_S / cc)
            else:
                SZ = proximal_operator_mixed_norm(
                    YSS, lambda_S / cc, rho_val=(1 - group)
                )

            SZ[abs(SZ) < 5e-4] = 0

            SZ_YS = SZ - YS
            LZ_YL = LZ - YL
            AZ_YA = np.dot(hrf, SZ_YS) + LZ_YL
            AZ = YA + AZ_YA

            S = SZ
            L = LZ
            A = AZ

            # Non zero voxel amount higher than vox_2_keep are considered low-rank.
            S_nonzero = np.count_nonzero(S, axis=1)
            global_fluc = np.where(S_nonzero > nvox * vox_2_keep)[0]
            S[global_fluc, :] = 0

            S_SO = S - SO
            L_LO = L - LO
            A_AO = A - AO

            t[i + 1] = (1 + np.sqrt(1 + 4 * t[i] ** 2)) / 2  # Combination parameter

            t1 = (t[i] - 1) / t[i + 1]

            YS = S + t1 * (S_SO)
            YL = L + t1 * (L_LO)
            YA = A + t1 * (A_AO)

            MSE_iter = np.sqrt(np.sum((np.dot(hrf, S) + L - data) ** 2, axis=0) / nt)

            if np.mean(abs(MSE_iter - noise_est)) <= tol:
                break

        l_final = L.copy()

    # Return eigen vectors we keep.
    Ut, St, Vt = svd(
        np.nan_to_num(l_final), full_matrices=False, compute_uv=True, check_finite=True
    )

    eig_vecs = Ut[:, :keep_idx]
    mean_eig_vecs = np.mean(eig_vecs, axis=0)
    std_eig_vecs = np.std(eig_vecs, axis=0)
    eig_vecs = (eig_vecs - mean_eig_vecs) / (std_eig_vecs)
    eig_maps = Vt[:keep_idx, :]
    mean_eig_maps = np.expand_dims(np.mean(eig_maps, axis=1), axis=1)
    std_eig_maps = np.expand_dims(np.std(eig_maps, axis=1), axis=1)
    eig_maps = (eig_maps - mean_eig_maps) / std_eig_maps

    return (l_final, S, eig_vecs, eig_maps)
