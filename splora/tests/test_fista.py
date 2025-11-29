"""Tests for FISTA solver."""

import os

import numpy as np
from pySPFM.deconvolution.select_lambda import select_lambda

from splora.deconvolution import fista
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(os.path.join(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(os.path.join(data_dir, "visual_task.1d"))
y_out = np.load(os.path.join(data_dir, "visual_task_output.npy"))


def test_proximal_operator_lasso():
    """Test soft-thresholding proximal operator."""
    a_all = np.array([1, -1, 2, -2, 0])
    a_thr = np.array([0, 0, 1, -1, 0])
    assert np.allclose(fista.proximal_operator_lasso(a_all, 1), a_thr)


def test_proximal_operator_mixed_norm():
    """Test L2,1 + L1 mixed-norm proximal operator."""
    a_all = np.array([[1, -1, 2, -2, 0], [1, -1, 2, -2, 0]])
    a_thr = np.array(
        [
            [0.17675047, -0.17675047, 1.06050283, -1.06050283, 0.0],
            [0.17675047, -0.17675047, 1.06050283, -1.06050283, 0.0],
        ]
    )
    assert np.allclose(fista.proximal_operator_mixed_norm(a_all, 1), a_thr)


def test_select_lambda():
    """Test lambda selection criteria from pySPFM."""
    # MAD Update and MAD - pySPFM returns lambda as the first value, not raw noise
    lambda_selec, update_lambda, noise = select_lambda(
        hrf=hrf, y=y, criterion="mad_update"
    )
    assert np.allclose(lambda_selec, 0.002248865844940937, rtol=1e-5)
    assert update_lambda
    assert np.allclose(
        noise, lambda_selec, rtol=1e-5
    )  # For mad_update, noise == lambda

    # Universal threshold
    lambda_selec, update_lambda, noise = select_lambda(hrf=hrf, y=y, criterion="ut")
    assert np.allclose(lambda_selec, 0.004721675779877547, rtol=1e-5)
    assert not update_lambda

    # Lower universal threshold
    lambda_selec, update_lambda, noise = select_lambda(hrf=hrf, y=y, criterion="lut")
    assert np.allclose(lambda_selec, 0.0041566221123978085, rtol=1e-5)
    assert not update_lambda

    # Factor
    lambda_selec, update_lambda, noise = select_lambda(
        hrf=hrf, y=y, criterion="factor", factor=10
    )
    assert not update_lambda

    # Percentage
    lambda_selec, update_lambda, noise = select_lambda(
        hrf=hrf, y=y, criterion="pcg", pcg=0.8
    )
    assert not update_lambda


def test_fista():
    """Test FISTA solver."""
    beta, _, _, noise_est, lambda_val, _ = fista.fista(
        hrf=hrf,
        y=np.expand_dims(y, axis=1),
        n_te=1,
        pfm_only=True,
        lambda_crit="factor",
        factor=20,
    )
    # Check output shape matches expected
    assert beta.shape == y_out.shape
    # Check that noise estimate and lambda are reasonable
    assert noise_est[0] > 0
    assert lambda_val[0] > 0
