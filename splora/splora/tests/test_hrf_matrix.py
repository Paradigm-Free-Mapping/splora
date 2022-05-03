import os

import numpy as np

from splora.deconvolution import hrf_matrix
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()


def test_hrf_linear():
    hrf_linear = np.load(os.path.join(data_dir, "hrf_linear.npy"))
    params = [6, 16, 1, 1, 6, 0, 32]
    hrf = hrf_matrix.hrf_linear(TR=2, p=params)
    assert np.allclose(hrf, hrf_linear)


def test_hrf_afni():
    hrf_afni = np.array(
        [
            0.0,
            0.0360894,
            0.156291,
            0.160475,
            0.0900993,
            0.0320469,
            0.00067546,
            -0.0127604,
            -0.0155529,
            -0.0128561,
            -0.00855318,
            -0.00485445,
            -0.00242662,
        ]
    )
    hrf = hrf_matrix.hrf_afni(TR=2)
    assert np.allclose(hrf, hrf_afni)


def test_HRFMatrix():
    hrf = np.load(os.path.join(data_dir, "hrf.npy"))
    hrf_norm = np.load(os.path.join(data_dir, "hrf_norm.npy"))
    hrf_obj = hrf_matrix.HRFMatrix(TE=[0]).generate_hrf()
    assert np.allclose(hrf_obj.X_hrf, hrf)
    assert np.allclose(hrf_obj.X_hrf_norm, hrf_norm)