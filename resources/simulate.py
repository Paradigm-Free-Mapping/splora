import numpy as np
import random
from low_rank_pfm.src.hrf_matrix import HRFMatrix


def _gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


class fMRIsim:
    def __init__(self,
                 event_length='mix',
                 nvoxels=1,
                 dur=400,
                 TR=2,
                 db=10,
                 nevents=1,
                 gap=5,
                 TE=None,
                 is_afni=True,
                 lop_hrf='SPMG1',
                 has_integrator=True,
                 tesla=3,
                 noise=True,
                 max_length=None,
                 min_length=None,
                 group=1,
                 ngroups=1,
                 motion=0.05):

        # Parameters for signal creation
        self.length = event_length
        self.nvoxels = nvoxels
        self.dur = dur
        self.tr = TR
        self.nevents = nevents
        self.gap = gap
        self.max_length = max_length
        self.min_length = min_length
        self.group = group
        self.ngroups = ngroups

        # Parameters for HRF matrix creation
        self.te = TE
        self.is_afni = is_afni
        self.lop_hrf = lop_hrf
        self.has_integrator = has_integrator

        # Noise
        self.tesla = tesla
        self.has_noise = noise
        self.db = db
        self.percent_motion = motion

    def _add_noise(self):
        # Noise due to motion
        self.percent_motion = 0.05  # percentage of noise that it is due to motion
        motion_par = np.random.randn(6, 1)
        regpar_mtx = np.genfromtxt('regparam.1D')
        regpar_mtx = regpar_mtx[0:self.nscans, :]
        regpar_mtx = _gram_schmidt_columns(regpar_mtx)

        # db simulations based on Tryantafyllou NI,2005 and van der Zwaag NI,2009
        # Relative signal change (DeltaS/S) at 1x1x3
        # (1.5T with TE=50ms = 0.02, 3T with TE=35=0.03, 7T with TE=25ms = 0.06)
        # Consider mean of signal = 1, then sigma_noise = 1/db
        self.S = 1
        if self.tesla is 1.5:
            self.DeltaStoS = 0.02
            par_sps0 = np.array([5.013e-6, 2.817, 0.397])
        elif self.tesla is 3:
            self.DeltaStoS = 0.04
            par_sps0 = np.array([6.42e-12, 5.962, 0.59])
        elif self.tesla is 7:
            self.DeltaStoS = 0.06
            par_sps0 = np.array([1.34e-14, 7.38, 0.9625])

        CNR = self.db * self.DeltaStoS
        # The relationship between physiological noise and thermal noise at
        # different voxels sizes
        # Taken from Triantafyllou at 1x1x3 (1.5T=0.34, 3T=0.57, 7T = 0.91)
        sigma_noise = self.S / self.db
        sps0 = par_sps0[0] * (np.power(self.db, par_sps0[1])) + par_sps0[2]
        sigma_thermalnoise = np.sqrt(
            (np.power(sigma_noise, 2)) / ((np.power(sps0, 2)) + 1))
        sigma_physionoise = sps0 * sigma_thermalnoise
        # Frequency of respiratory fluctuations
        fcardiac = 1.1  # cardiac frequency (Hertz) (Shmueli NI2007)
        fresp = 0.3  # respiratory frequency (Hertz) (Birn NI2006, Glover MRM2000)
        n_harmonics = 2
        power_harmonics = 2
        # from Kruger and Glover MRM 2001, physiological noise is composed arises
        # from fluctuations in the basal CMRO2, CBF, CBV, but also cardiac and
        # respiratory functions that cause quasiperiodic oscillations in the
        # vascular system AND motion from subtle brain pulsability.
        # So sigma_physionoise is divided in two terms: physiological sinusoidal
        # fluctuations and motion.
        sigma_motion = np.sqrt(self.percent_motion) * sigma_physionoise
        sigma_physionoise = np.sqrt(1 - self.percent_motion) * sigma_physionoise

        # generate thermal noise signals
        n_thermal = np.random.randn(self.nscans, 1)
        n_thermal = sigma_thermalnoise * n_thermal / np.std(n_thermal)
        # generate physiological noise signals
        s_cardiac = (np.power(
            power_harmonics,
            0)) * np.sin(2 * np.pi * fcardiac * self.tr *
                         np.linspace(0, self.nscans - 1, self.nscans) +
                         2 * np.pi * np.random.rand(1)[0])
        s_resp = (np.power(
            power_harmonics,
            0)) * np.sin(2 * np.pi * fresp * self.tr *
                         np.linspace(0, self.nscans - 1, self.nscans) +
                         2 * np.pi * np.random.rand(1)[0])
        for j in range(n_harmonics - 1):
            s_cardiac = s_cardiac + (np.power(
                power_harmonics,
                (-j))) * np.sin(2 * np.pi * (j + 1) * fcardiac * self.tr *
                                np.linspace(0, self.nscans - 1, self.nscans) +
                                2 * np.pi * np.random.rand(1)[0])
            s_resp = s_resp + (np.power(
                power_harmonics,
                (-j))) * np.sin(2 * np.pi * (j + 1) * fresp * self.tr *
                                np.linspace(0, self.nscans - 1, self.nscans) +
                                2 * np.pi * np.random.rand(1)[0])

        s_cardiac = np.expand_dims(s_cardiac, axis=-1)
        s_resp = np.expand_dims(s_resp, axis=-1)

        n_physio = s_resp + s_cardiac
        n_physio = sigma_physionoise * n_physio / np.std(n_physio)
        n_motion = regpar_mtx.dot(motion_par)
        n_motion = sigma_motion * n_motion / np.std(n_motion)
        # Append random motion values when requested number of scans is larger than 1D file
        if n_motion.shape[0] < self.nscans:
            diff = self.nscans - n_motion.shape[0]
            idxs = np.random.randint(low=0, high=n_motion.shape[0], size=diff)
            n_motion = np.append(n_motion, n_motion[idxs, :], 0)

        # Sum of noise from different sources
        noise = n_physio + n_thermal + n_motion

        # # Concatenate noise for multi echo design
        # if len(self.te) > 1:
        #     temp = noise.copy()
        #     for i in range(len(self.te) - 1):
        #         noise = np.append(noise, temp, 0)

        return noise

    def simulate(self):

        # Variable initialization
        self.te = np.asarray(self.te)
        self.te = self.te / 1000
        self.nscans = int(self.dur / self.tr)
        self.simulation = np.zeros((self.nscans * len(self.te), self.nvoxels))
        self.bold = self.simulation.copy()
        self.r2 = np.zeros((self.nscans, self.nvoxels))
        self.innovation = self.r2.copy()
        self.noise = self.bold.copy()

        if self.length == 'mix':
            ev_list = ['short', 'medium', 'long']

        group_change_idxs = np.zeros((self.ngroups + 1, ))
        group_change_idxs[1:] = np.cumsum(self.group) - 1
        group_label = 0

        print(f'Groups: {self.group}')
        print(f'Group change idxs: {group_change_idxs}')

        for voxidx in range(self.nvoxels):
            if group_label < len(group_change_idxs) - 1:
                if voxidx == group_change_idxs[group_label]:
                    idx_avail = np.arange(2, self.nscans)
                    group_label += 1
                    for eventidx in range(self.nevents):
                        if self.max_length and self.min_length:
                            temp_len = np.random.randint(self.min_length,
                                                        self.max_length)
                            temp_pos = int(
                                random.choice(idx_avail[:len(idx_avail) -
                                                        temp_len]))
                            self.r2[temp_pos:temp_pos + temp_len, voxidx] = 1
                            idx_pos = np.where(idx_avail == temp_pos)[0]
                            idx_sel = np.arange(idx_pos - 2,
                                                idx_pos + temp_len + 2)
                            np.delete(idx_avail, idx_sel)
                        else:
                            if self.length == 'mix':
                                ev_len = random.choice(ev_list)
                            else:
                                ev_len = self.length

                            if ev_len == 'short':  # Event length [1,1%]
                                if self.max_length is None:
                                    if np.ceil(0.01 * self.nscans) > 1:
                                        self.max_length = 0.01 * self.nscans
                                    else:
                                        self.max_length = 2
                                temp_len = np.random.randint(1, self.max_length)
                                temp_pos = int(
                                    random.choice(idx_avail[:len(idx_avail) -
                                                            temp_len]))
                                self.r2[temp_pos:temp_pos + temp_len, voxidx] = 1
                                idx_pos = np.where(idx_avail == temp_pos)[0]
                                idx_sel = np.arange(idx_pos - 2,
                                                    idx_pos + temp_len + 2)
                                np.delete(idx_avail, idx_sel)
                            elif ev_len == 'medium':  # Event length [1%, 5%]
                                if self.max_length is None:
                                    self.max_length = np.ceil(0.1 * self.nscans)
                                temp_len = np.random.randint(
                                    np.ceil(0.05 * self.nscans), self.max_length)
                                temp_pos = int(
                                    random.choice(idx_avail[:len(idx_avail) -
                                                            temp_len]))
                                idx_pos = np.where(idx_avail == temp_pos)[0]
                                while (idx_avail[idx_pos + temp_len] -
                                    idx_avail[idx_pos]) > temp_len:
                                    temp_len -= 1
                                self.r2[temp_pos:temp_pos + temp_len, voxidx] = 1
                                idx_sel = np.arange(idx_pos - 2,
                                                    idx_pos + temp_len + 2)
                                np.delete(idx_avail, idx_sel)
                            elif ev_len == 'long':  # Event length [5% 10%]
                                if self.max_length is None:
                                    self.max_length = np.ceil(0.1 * self.nscans)
                                temp_len = np.random.randint(
                                    np.ceil(0.05 * self.nscans), self.max_length)
                                temp_pos = int(
                                    random.choice(idx_avail[:len(idx_avail) -
                                                            temp_len]))
                                idx_pos = np.where(idx_avail == temp_pos)[0]
                                while (idx_avail[idx_pos + temp_len] -
                                    idx_avail[idx_pos]) > temp_len:
                                    temp_len -= 1
                                self.r2[temp_pos:temp_pos + temp_len, voxidx] = 1
                                idx_sel = np.arange(idx_pos - 2,
                                                    idx_pos + temp_len + 2)
                                np.delete(idx_avail, idx_sel)
                else:
                    self.r2[:, voxidx] = self.r2[:, voxidx - 1].copy()

            # print('Voxel {}/{} simulated...'.format(voxidx + 1, self.nvoxels))

        if group_change_idxs[1] - group_change_idxs[0] == 1:
            self.r2[:, 0] = self.r2[:, 1].copy()

        print('Finishing simulation of data...')

        # Gets innovation signal
        self.innovation[:self.nscans - 1, :] = self.r2[1:, :]
        self.innovation = self.innovation - self.r2

        # Generate HRF matrix
        hrf_matrix = HRFMatrix(TR=self.tr,
                               TE=self.te,
                               nscans=self.nscans,
                               r2only=True,
                               is_afni=self.is_afni,
                               lop_hrf=self.lop_hrf)
        hrf_matrix.generate_hrf()
        self.hrf = hrf_matrix.X_hrf
        self.hrf_norm = hrf_matrix.X_hrf_norm

        # Positive changes in BOLD are generated by negative changes in R2* in multi echo
        if len(self.te) > 1:
            self.innovation = -self.innovation
            self.r2 = -self.r2

        # Convolve HRF with generated R2* signals to get BOLD timeseries
        if self.has_integrator:
            self.bold = np.dot(self.hrf, self.innovation)
        else:
            self.bold = np.dot(self.hrf, self.r2)

        for te_idx in range(len(self.te)):
            temp_bold = self.bold[te_idx * self.nscans:(te_idx + 1) *
                                  self.nscans, :].copy()

            # Noise
            if self.has_noise:
                temp_noise = self._add_noise()
            else:
                temp_noise = 0

            self.noise[te_idx * self.nscans:(te_idx + 1) *
                       self.nscans, :] = temp_noise

            if self.has_noise:
                psio = np.sum((temp_bold)**2)/temp_bold.shape[0]
                # noisevar = np.sqrt(psio*temp_bold.shape[0])/np.sqrt(np.sum(temp_noise**2))/np.sqrt(10**(self.db))
                for voxidx in range(self.nvoxels):
                    self.simulation[
                        te_idx * self.nscans:(te_idx + 1) * self.
                        nscans, voxidx] = np.squeeze(temp_bold[:, voxidx]) + np.squeeze(temp_noise * np.random.rand())
            else:
                self.simulation[
                    te_idx * self.nscans:(te_idx + 1) * self.
                    nscans, :] = temp_bold

            del temp_noise, temp_bold
