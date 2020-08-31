import numpy as np
import time
from scipy.linalg import svd
from scipy.stats import median_absolute_deviation
from pywt import wavedec
from scipy import stats


# soft-thresholding function
def SoftThresh(x, p, is_low_rank=False):
    xa = abs(x)
    xap = (xa > p)
    y = (xa - p) * (x / xa) * xap
    y[np.invert(xap)] = 0
    return(y)


def low_rank(data, hrf, maxiter=1000, miniter=10, vox_2_keep=0.3):
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

    print('MFISTA-AS for L2-L+S problems')

    nt = data.shape[0]
    nvox = data.shape[1]

    _, cD1 = wavedec(data, 'db3', level=1, axis=0)

    noise_est = stats.median_absolute_deviation(cD1) / 0.6745
    nv_2_save = np.zeros((nvox, 50))
    L = np.zeros((nt, nvox))
    S = np.zeros((nt, nvox))

    # algorithm parameters
    cc = 10
    mu_in = 1.5
    tol = 1e-6
    restart = False
    comp_cost = True
    display = True

    # iterations
    converged = False
    l_iter = 0

    # lambda_S = noise_est * np.sqrt(2*np.log10(nt))
    # lambda_S = noise_est * np.sqrt(2*np.log10(nt) - np.log10(1 + 4 * np.log10(nt)))

    for ii in range(2):

        data[abs(data) < 1e-3] = 0

        i = 0

        if l_iter == 0:
            A = np.dot(hrf, S) + L
            A_lambda = np.dot(hrf, S)
        # else:
            # A = np.dot(hrf, S_deb) + L
            # A_lambda = np.dot(hrf, S_deb)
        L_old = L.copy()
        YL = L.copy()
        YS = S.copy()
        YA = A.copy()

        # Initializes cost arrays
        L2cost = np.zeros((maxiter, ))
        L1cost = np.zeros((maxiter, ))
        Ncost = np.zeros((maxiter, ))
        COST = np.zeros((maxiter, ))
        ERR = np.zeros((maxiter, ))
        MDIF = np.zeros((maxiter, ))
        x_diff = np.zeros((maxiter, ))
        t = np.zeros((maxiter, ))
        zeta = np.zeros((maxiter, ))
        eta = np.zeros((maxiter, ))
        delta = np.zeros((maxiter, ))
        TIM = np.zeros((maxiter, ))
        mu = np.zeros((maxiter, ))

        start_time = time.time()
        TIM[i] = time.time() - start_time

        # if l_iter == 0:
        Ut, St, Vt = svd(data, full_matrices=False,
                             compute_uv=True, check_finite=True)
        # else:
        #     Ut, St, Vt = svd(A, full_matrices=False,
        #                      compute_uv=True, check_finite=True)
        St = np.diag(St)
        if l_iter == 0:
            lambda_S = noise_est
        # else:
        #     _, cD1 = wavedec(data, 'db3', level=1, axis=0)
        #     lambda_S = stats.median_absolute_deviation(cD1) / 0.6745

        non_noisy = St[St > 3*np.std(St)]
        mad = median_absolute_deviation(non_noisy)
        # print(non_noisy)
        print(f'Median: {np.median(non_noisy)} and MAD: {mad}')
        keep_idx = len(St[St > (np.median(non_noisy) + mad)])
        print(f'Keeping {keep_idx} eigenvalues...')
        # if np.diag(St)[keep_idx] != 0:
        #     lambda_L = np.diag(St)[keep_idx] * 1.01
        # else:
        lambda_L = np.diag(St)[keep_idx] * 1.01
        nv = np.ones((nvox, ))

        # else:
        #     # nv = np.sqrt(np.sum((A_lambda - data) ** 2, axis=0) / nt)
        #     # if(abs(nv - noise_est).all() > tol):
        #     #     lambda_S = lambda_S*noise_est / nv
        #     # _, cD1 = wavedec(data, 'db3', level=1, axis=0)

        #     # noise_est = stats.median_absolute_deviation(cD1) / 0.6745
        #     # lambda_S = noise_est * 1.1
        #     print(MSE_iter)
        #     print(noise_est)
        #     # if abs(ERR[i] - noise_est).all() > tol:
        #     #     lambda_S = lambda_S * noise_est / MSE_iter

        nv_2_save[:, l_iter] = nv

        print(f'NV: {nv}')
        print(f'Lambda S: {lambda_S}')

        St[keep_idx:] = 0
        # if l_iter > 0:
        #     rho = rho*0.9995
        #     print(f'Rho: {rho}')
        # if rho <= 0.99:
        #     break
        # lambda_val = lambda_L/rho
        # lambda_S = lambda_val * (1 - rho)
        # print(f'Lambda: {lambda_S}')

        # Residue
        L2cost[i] = (1 / 2
                    * np.linalg.norm(data.flatten() - A.flatten(), ord=2) ** 2)
        L1cost[i] = np.linalg.norm(S.flatten(), ord=1)
        Ls = svd(L, full_matrices=False, compute_uv=False)
        Ncost[i] = np.sum(Ls)
        COST[i] = L2cost[i] + lambda_L * Ncost[i] + np.mean(lambda_S) * L1cost[i]
        # Estimation error
        ERR[i] = np.linalg.norm(data.flatten() - A.flatten(), ord=2)

        MDIF[i] = 1e20
        x_diff[i] = MDIF[i]
        ncDIF = 0
        t[i] = 1
        convergence_criteria = 1

        for i in range(100): #((i < 100)): #and ((MDIF[i] >= tol) or ncDIF)):
            # data consistency gradient
            y_YA = data - YA

            LO = L.copy()
            SO = S.copy()
            AO = A.copy()

            # Low-rank update
            if(lambda_L != 0):
                Ut, St, Vt = svd(np.nan_to_num(YL+(1/cc)*y_YA), full_matrices=False,
                                 compute_uv=True, check_finite=True)
                St = np.diag(SoftThresh(St, lambda_L/cc, is_low_rank=True))
                LZ = np.dot(np.dot(Ut, St), Vt)
            else:
                LZ = np.zeros((L.shape))
                YL = np.zeros((L.shape))
            
            # if i > 0 and abs(ERR[i] - noise_est).all() > tol:
            #     lambda_S = lambda_S * noise_est / ERR[i]

            # Sparse update
            # if(lambda_S != 0):
            YSS = YS + (1 / cc) * y_YA

            SZ = SoftThresh(YSS, lambda_S / cc)
            # else:
            #     S = np.zeros((S.shape))
            #     SZ = np.zeros((S.shape))
            #     YS = np.zeros((S.shape))

            SZ_YS = SZ - YS
            LZ_YL = LZ - YL
            AZ_YA = np.dot(hrf, SZ_YS) + LZ_YL
            AZ = YA + AZ_YA

            dA = AZ - AO
            dS = SZ - SO
            dL = LZ-LO

            # Majorizer gap
            y_AZ = data - AZ
            f_Z = .5*(np.dot(y_AZ.flatten().T, y_AZ.flatten()))
            f_Y = .5*(np.dot(y_YA.flatten().T, y_YA.flatten()))
            QdZY = ((cc / 2 * np.linalg.norm(LZ_YL.flatten(), ord=2) ** 2)
                    + (cc / 2 * np.linalg.norm(SZ_YS.flatten(), ord=2) ** 2))
            zeta[i] = (f_Y - np.real(np.dot(y_YA.flatten().T, SZ_YS.flatten())
                    + np.dot(y_YA.flatten().T, LZ_YL.flatten()))
                    + QdZY - f_Z)

            LZs = svd(LZ, full_matrices=False,
                      compute_uv=False, check_finite=True)
            COSTCZ = (f_Z + lambda_L * sum(LZs) + np.mean(lambda_S)
                    * np.linalg.norm(SZ.flatten(), ord=1))

            if(COSTCZ < COST[i]):
                S = SZ
                L = LZ
                A = AZ
                ncDIF = 0
                COSTC = COSTCZ
                mu[i] = 1
            else:
                S = SO
                L = LO
                A = AO
                ncDIF = 0
                COSTC = COST[i]
                mu[i] = 0

            if(mu_in != 1):
                SS = SO + mu_in * dS
                LS = LO + mu_in * dL
                AS = AO + mu_in * dA

                y_AS = data - AS
                LSs = svd(LS, full_matrices=False,
                          compute_uv=False, check_finite=True)
                COSTCS = (np.dot(y_AS.flatten().T, y_AS.flatten()) / 2
                        + lambda_L * np.sum(LSs) + np.mean(lambda_S)
                        * np.linalg.norm(SS.flatten(), ord=1))

                if COSTCS < COSTCZ:
                    if COSTCS < COST[i]:
                        S = SS
                        L = LS
                        A = AS
                        COSTC = COSTCS
                        mu[i] = mu_in
            S_nonzero = np.count_nonzero(S, axis=1)
            global_fluc = np.where(S_nonzero > nvox * vox_2_keep)[0]
            S[global_fluc, :] = 0

            # Alpha step gap
            delta[i] = -COSTC + COSTCZ

            # Overstep
            eta[i] = 1 + (zeta[i] + delta[i]) / (QdZY + np.finfo(float).eps)

            S_SO = S - SO
            L_LO = L - LO
            A_AO = A - AO
            SZ_S = SZ - S
            LZ_L = LZ - L
            AZ_A = AZ - A

            # Restart is experimental
            rest1 = (mu[i] == 0)
            if((rest1) and restart):
                t[i] = 1
                SZ_S = 0
                LZ_L = 0
                AZ_A = 0

            t[i+1] = (1 + np.sqrt(1 + 4 * t[i] ** 2)) / 2  # Combination parameter

            t1 = (t[i] - 1) / t[i+1]
            t2 = t[i] / t[i+1]
            t3 = (t[i] / t[i+1]) * (eta[i] - 1)

            YS = S + t1 * (S_SO) + t2 * (SZ_S) + t3 * (SZ_YS)
            YL = L + t1 * (L_LO) + t2 * (LZ_L) + t3 * (LZ_YL)
            YA = A + t1 * (A_AO) + t2 * (AZ_A) + t3 * (AZ_YA)

            i += 1

            TIM[i] = time.time() - start_time

            y_A = data - A

            if(comp_cost):
                L2cost[i] = np.dot(y_A.flatten().T, y_A.flatten()) / 2  # Residue
                L1cost[i] = np.linalg.norm(S.flatten(), ord=1)
                Ls = svd(L, full_matrices=False,
                         compute_uv=False, check_finite=True)
                Ncost[i] = np.sum(Ls)
                COST[i] = L2cost[i] + lambda_L * Ncost[i] + np.mean(lambda_S) * L1cost[i]
                ERR[i] = np.linalg.norm(data.flatten() - A.flatten(), ord=2)/nt
                # Print some numbers
                if display:
                    print(f'mfista-va i={i}, cost={COST[i]:.9f},'
                          f'err={ERR[i]:.9f}, L={cc:.3f}, mu={mu[i-1]:.3f}, ')
                    print(f'delta={delta[i-1]:.3f}, zeta={zeta[i-1]:.3f}, '
                          f'eta={eta[i-1]:.3f}  \n')

            else:
                COST[i] = COSTC
                if display:
                    print(f'mfista-va i={i}, L={cc:.3f}, mu={mu[i-1]:.3f}, ')
                    print(f'delta={delta[i-1]:.3f}, zeta={zeta[i-1]:.3f}, '
                          f'eta={eta[i-1]:.3f}  \n')

            # x_diff[i+1] = (np.linalg.norm(A.flatten() - AO.flatten())
            #                / np.linalg.norm(A.flatten()))

            # Force at least 10 itereations with no improvement
            ii = np.min((i+1, 10)) - 1
            if (i-ii) == 0:
                MDIF[i] = np.max(x_diff[i::-1])
            else:
                MDIF[i] = np.max(x_diff[i:i-ii-1:-1])

            if i > miniter and (ERR[i] - ERR[i-1]) < tol:
                break

            # convergence_criteria = np.power((S - SO), 2).sum()/np.power(SO, 2).sum()
        # END WHILE

        # deb_out = debiasing(hrf, data, S)
        # S_deb = deb_out['beta']

        MSE_iter = np.min(np.sqrt(np.sum(abs(((np.dot(hrf, S) + L) - data)) ** 2, axis=0)) / nt)

        print(f'MSE on iter {l_iter+1} is {MSE_iter}')
        if l_iter == 0:
            MSE = MSE_iter
            counter = 1
        else:
            MSE = np.hstack((MSE, MSE_iter))
            if MSE[l_iter] < tol:
                converged = True
                print('FISTA has converged!!!')

        if (l_iter > 0) and (MSE[l_iter - 1] == MSE[l_iter]):
            counter += 1

        if (l_iter > 0) and (np.abs(MSE[l_iter-1] - MSE[l_iter]) <= tol) and (MSE[l_iter - 1] > MSE[l_iter]):
            converged = True
            print('MSE not improving!!!')
            break

        if (l_iter > 0) and (MSE[l_iter - 1] < MSE[l_iter]):
            counter += 1

        if counter == 5:
            converged = True
            print('MSE not improving!!!')
            break

        if l_iter == 0:
            l_final = L.copy()
        else:
            l_final = L + L_old
            # break

        # plt.figure(figsize=(16, 9))
        # plt.plot(orig_data, label='Sim', color='#000000', linewidth=2)
        # plt.plot(noise[:, 0], label='Sim L', color='#246d9e', linewidth=1.2)
        # plt.plot(l_final[:, 0], label='L', color='#38fcae',
        #          linestyle='-', linewidth=0.7)
        # plt.plot(sim[:, 0], label='Sim S', color='#ba5c29', linewidth=1.2)
        # plt.plot(S[:, 0], label='S', color='red',
        #          linestyle='-', linewidth=0.7)
        # plt.legend()
        # plt.show()
        # plt.clf()

        # # Y
        # _, Y_St, _ = svd(data, full_matrices=False, compute_uv=True, check_finite=True)
        # fig = plt.figure(figsize=(16, 9))
        # plt.plot(np.diag(Y_St)[:20], '-o', label='Y')
        # _, YL_St, _ = svd(data-L, full_matrices=False,
        #                   compute_uv=True, check_finite=True)
        # plt.plot(np.diag(YL_St)[:20], '-o', label='Y-L')
        # _, YHS_St, _ = svd(data-np.dot(hrf, S), full_matrices=False,
        #                    compute_uv=True, check_finite=True)
        # plt.plot(np.diag(YHS_St)[:20], '-o', label='Y-HS')
        # plt.savefig(f'y_eigvals_{l_iter}.png', dpi=300)
        # plt.close(fig)

        l_iter += 1
        if l_iter == 0:
            break
        MSE_iter = np.sqrt(np.sum(((np.dot(hrf, S) + L) - data) ** 2, axis=0) / nt)
        data = data - L
    # END WHILE

    S_nonzero = np.count_nonzero(S, axis=1)
    global_fluc = np.where(S_nonzero > nvox * vox_2_keep)[0]
    S[global_fluc, :] = 0

    return(l_final, S)

    # L=reshape(L,nx,ny,nt);
    # S=reshape(param.W'*WS,nx,ny,nt);

    # recon.im=L+S;
    # recon.L=L;
    # recon.S=S;
    # recon.WS=WS;
    # recon.lambda_S = param.lambda_S;
    # recon.lambda_L = param.lambda_L;
    # recon.err=ERR;
    # recon.diff=x_diff;
    # recon.nite=i-1;
    # recon.L2=L2cost;
    # recon.L1=L1cost;
    # recon.N=Ncost;
    # recon.cost=COST;
    # recon.tim=TIM;
    # recon.c=c;
    # recon.mu=mu;
    # recon.eta=eta;
    # recon.zeta=zeta;
    # recon.delta=delta;
