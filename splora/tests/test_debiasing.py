import numpy as np
import os

from splora.deconvolution import debiasing
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(os.path.join(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(os.path.join(data_dir, "visual_task.1d"))
y_out = np.load(os.path.join(data_dir, "visual_task_output.npy"))


def test_debiasing_block():
    beta_check = np.load(os.path.join(data_dir, "deb_block.npy"))
    beta = debiasing.debiasing_block(hrf=hrf, y=np.expand_dims(y, axis=1), auc=y_out)
    assert np.allclose(beta, beta_check)


def test_debiasing_spike():
    beta_check = np.load(os.path.join(data_dir, "deb_spike_beta.npy"))
    fitted_check = np.load(os.path.join(data_dir, "deb_spike_fitted.npy"))
    beta, fitted = debiasing.debiasing_spike(hrf=hrf, y=np.expand_dims(y, axis=1), auc=y_out)
    assert np.allclose(beta, beta_check)
    assert np.allclose(fitted, fitted_check)
