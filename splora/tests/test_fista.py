import os

import numpy as np

from splora.deconvolution import fista
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(os.path.join(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(os.path.join(data_dir, "visual_task.1d"))
y_out = np.load(os.path.join(data_dir, "visual_task_output.npy"))


def test_proximal_operator_lasso():
    a_all = np.array([1, -1, 2, -2, 0])
    a_thr = np.array([0, 0, 1, -1, 0])
    assert np.allclose(fista.proximal_operator_lasso(a_all, 1), a_thr)


def test_proximal_operator_mixed_norm():
    a_all = np.array([[1, -1, 2, -2, 0], [1, -1, 2, -2, 0]])
    a_thr = np.array(
        [
            [0.17675047, -0.17675047, 1.06050283, -1.06050283, 0.0],
            [0.17675047, -0.17675047, 1.06050283, -1.06050283, 0.0],
        ]
    )
    assert np.allclose(fista.proximal_operator_mixed_norm(a_all, 1), a_thr)


def test_select_lambda():
    noise_expected = 0.0015168392317151877
    # MAD Update and MAD
    lambda_selec, update_lambda, noise = fista.select_lambda(hrf=hrf, y=y, criteria="mad_update")
    assert np.allclose(lambda_selec, noise_expected)
    assert update_lambda
    assert np.allclose(noise, noise_expected)

    # Universal threshold
    lambda_selec, update_lambda, noise = fista.select_lambda(hrf=hrf, y=y, criteria="ut")
    assert np.allclose(lambda_selec, 0.0031847266827718513)
    assert not update_lambda
    assert np.allclose(noise, noise_expected)

    # Lower universal threshold
    lambda_selec, update_lambda, noise = fista.select_lambda(hrf=hrf, y=y, criteria="lut")
    assert np.allclose(lambda_selec, 0.002803603205448407)
    assert not update_lambda
    assert np.allclose(noise, noise_expected)

    # Factor
    lambda_selec, update_lambda, noise = fista.select_lambda(
        hrf=hrf, y=y, criteria="factor", factor=10
    )
    assert np.allclose(lambda_selec, 0.015168392317151877)
    assert not update_lambda
    assert np.allclose(noise, noise_expected)

    # Percentage
    lambda_selec, update_lambda, noise = fista.select_lambda(hrf=hrf, y=y, criteria="pcg", pcg=0.8)
    assert np.allclose(lambda_selec, 0.07034059283844483)
    assert not update_lambda
    assert np.allclose(noise, noise_expected)


def test_fista():
    beta, _, _, noise_est, lambda_val, _ = fista.fista(
        hrf=hrf,
        y=np.expand_dims(y, axis=1),
        n_te=1,
        pfm_only=True,
        lambda_crit="factor",
        factor=20,
    )
    assert np.allclose(beta, y_out, atol=1e-5)
    assert noise_est[0] == 0.0015168392317151877
    assert lambda_val[0] == 0.030336784634303754
