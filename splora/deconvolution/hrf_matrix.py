"""HRF Matrix file."""
import logging
import subprocess

import numpy as np
import scipy.io
import scipy.stats

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def hrf_linear(TR, p):
    """Generate HRF.

    Parameters
    ----------
    TR : float
        TR of the acquisition.
    p : list
        Input parameters of the response function (two gamma functions).
                                                          defaults
                                                         (seconds)
        p[1] - delay of response (relative to onset)         6
        p[2] - delay of undershoot (relative to onset)      16
        p[3] - dispersion of response                        1
        p[4] - dispersion of undershoot                      1
        p[5] - ratio of response to undershoot               6
        p[6] - onset (seconds)                               0
        p[7] - length of kernel (seconds)                   32

    Returns
    -------
    hrf : array_like
        A hemodynamic response function (HRF).

    Notes
    -----
    Based on the spm_hrf function in SPM8
    Written in R by Cesar Caballero Gaudes
    Translated into Python by Eneko Urunuela
    """
    # global parameter
    # --------------------------------------------------------------------------
    fMRI_T = 16

    # modelled hemodynamic response function - {mixture of Gammas}
    # --------------------------------------------------------------------------
    dt = TR / fMRI_T
    u = np.arange(0, p[6] / dt + 1, 1) - p[5] / dt
    a1 = p[0] / p[2]
    b1 = 1 / p[3]
    a2 = p[1] / p[3]
    b2 = 1 / p[3]

    hrf = (
        scipy.stats.gamma.pdf(u * dt, a1, scale=b1)
        - scipy.stats.gamma.pdf(u * dt, a2, scale=b2) / p[4]
    ) / dt
    time_axis = np.arange(0, int(p[6] / TR + 1), 1) * fMRI_T
    hrf = hrf[time_axis]
    min_hrf = 1e-9 * min(hrf[hrf > 10 * np.finfo(float).eps])

    if min_hrf < 10 * np.finfo(float).eps:
        min_hrf = 10 * np.finfo(float).eps

    hrf[hrf == 0] = min_hrf

    return hrf


def hrf_afni(TR, lop_hrf="SPMG1"):
    """Generate HRF with AFNI's 3dDeconvolve.

    Parameters
    ----------
    TR : float
        TR of the acquisition.
    lop_hrf : str
        3dDeconvolve option to select HRF shape, by default "SPMG1"

    Returns
    -------
    hrf : array_like
        A hemodynamic response function (HRF).

    Notes
    -----
    AFNI installation is needed as it runs 3dDeconvolve on the terminal wtih subprocess.
    """
    dur_hrf = 8
    last_hrf_sample = 1
    # Increases duration until last HRF sample is zero
    while last_hrf_sample != 0:
        dur_hrf = 2 * dur_hrf
        hrf_command = (
            "3dDeconvolve -x1D_stop -nodata %d %f -polort -1 -num_stimts 1 -stim_times 1 "
            "'1D:0' '%s' -quiet -x1D stdout: | 1deval -a stdin: -expr 'a'"
        ) % (dur_hrf, TR, lop_hrf)
        hrf_tr_str = subprocess.check_output(
            hrf_command, shell=True, universal_newlines=True
        ).splitlines()
        hrf = np.array([float(i) for i in hrf_tr_str])
        last_hrf_sample = hrf[len(hrf) - 1]
        if last_hrf_sample != 0:
            LGR.info(
                "Duration of HRF was not sufficient for specified model. Doubling duration "
                "and computing again."
            )

    # Removes tail of zero samples
    while last_hrf_sample == 0:
        hrf = hrf[0 : len(hrf) - 1]
        last_hrf_sample = hrf[len(hrf) - 1]

    return hrf


class HRFMatrix:
    """A class for generating an HRF matrix.

    Parameters
    ----------
    TR : float
        TR of the acquisition, by default 2
    TE : list
        Values of TE in ms, by default None
    nscans : int
        Number of volumes in acquisition, by default 200
    r2only : bool
        Whether to only consider R2* in the signal model, by default True
    is_afni : bool
        Whether to use AFNI's 3dDeconvolve to generate HRF matrix, by default True
    lop_hrf : str
        3dDeconvolve option to select HRF shape, by default "SPMG1"
    block : bool
        Whether to use the block model in favor of the spike model, by default false
    """

    def __init__(
        self,
        TR=2,
        TE=None,
        nscans=200,
        r2only=True,
        is_afni=True,
        lop_hrf="SPMG1",
        block=True,
    ):
        self.TR = TR
        self.TE = TE
        self.nscans = nscans
        self.r2only = r2only
        self.lop_hrf = lop_hrf
        self.is_afni = is_afni
        self.block = block

    def generate_hrf(self):
        """Generate HRF matrix.

        Returns
        -------
        self
        """
        if self.is_afni:
            hrf_SPM = hrf_afni(self.TR, self.lop_hrf)
        else:
            p = [6, 16, 1, 1, 6, 0, 32]
            hrf_SPM = hrf_linear(self.TR, p)

        self.L_hrf = len(hrf_SPM)  # Length
        max_hrf = max(abs(hrf_SPM))  # Max value
        filler = np.zeros(self.nscans - hrf_SPM.shape[0], dtype=np.int)
        hrf_SPM = np.append(hrf_SPM, filler)  # Fill up array with zeros until nscans

        temp = hrf_SPM

        for i in range(self.nscans - 1):
            foo = np.append(np.zeros(i + 1), hrf_SPM[0 : (len(hrf_SPM) - i - 1)])
            temp = np.column_stack((temp, foo))

        if len(self.TE) > 1:
            tempTE = -self.TE[0] * temp
            for teidx in range(len(self.TE) - 1):
                tempTE = np.vstack((tempTE, -self.TE[teidx + 1] * temp))
        else:
            tempTE = temp

        if self.r2only:
            self.X_hrf = tempTE

        self.X_hrf_norm = self.X_hrf / max_hrf

        if self.block:
            if len(self.TE) > 1:
                for teidx in range(len(self.TE)):
                    temp = self.X_hrf[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ].copy()
                    self.X_hrf[teidx * self.nscans : (teidx + 1) * self.nscans - 1, :] = np.dot(
                        temp, np.tril(np.ones(self.nscans))
                    )
                    temp = self.X_hrf_norm[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ].copy()
                    self.X_hrf_norm[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ] = np.dot(temp, np.tril(np.ones(self.nscans)))
            else:
                self.X_hrf = np.dot(self.X_hrf, np.tril(np.ones(self.nscans)))
                self.X_hrf_norm = np.dot(self.X_hrf_norm, np.tril(np.ones(self.nscans)))

        return self
