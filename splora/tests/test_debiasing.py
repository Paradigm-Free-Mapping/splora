from os.path import join as opj

import numpy as np

from splora.deconvolution import debiasing
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(opj(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(opj(data_dir, "visual_task.1d"))
y_out = np.load(opj(data_dir, "visual_task_output.npy"))


def test_innovation_to_block():
    beta_check = np.load(opj(data_dir, "inno.npy"))
    beta, S = debiasing.innovation_to_block(hrf=hrf, y=y, auc=y_out, is_ls=True)
    assert S.shape == (160, 105)
    assert np.allclose(beta_check, beta)


def test_do_debias_block():
    beta_check = np.load(opj(data_dir, "deb_block.npy"))
    beta = debiasing.do_debias_block(hrf=hrf, y=np.expand_dims(y, axis=1), auc=np.squeeze(y_out))
    assert np.allclose(np.squeeze(beta_check), beta, atol=1e-5)


def test_debiasing_block():
    beta_check = np.load(opj(data_dir, "deb_block.npy"))
    beta = debiasing.debiasing_block(hrf=hrf, y=np.expand_dims(y, axis=1), auc=y_out)
    assert np.allclose(beta, beta_check)


def test_do_debias_spike():
    beta_check = np.load(opj(data_dir, "deb_spike_beta.npy"))
    fitted_check = np.load(opj(data_dir, "deb_spike_fitted.npy"))
    beta, fitted = debiasing.do_debias_spike(hrf=hrf, y=y, auc=y_out)
    assert np.allclose(beta, np.squeeze(beta_check))
    assert np.allclose(fitted, np.squeeze(fitted_check))


def test_debiasing_spike():
    beta_check = np.load(opj(data_dir, "deb_spike_beta.npy"))
    fitted_check = np.load(opj(data_dir, "deb_spike_fitted.npy"))
    beta, fitted = debiasing.debiasing_spike(hrf=hrf, y=np.expand_dims(y, axis=1), auc=y_out)
    assert np.allclose(beta, beta_check)
    assert np.allclose(fitted, fitted_check)
