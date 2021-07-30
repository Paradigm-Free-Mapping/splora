import numpy as np

from splora.deconvolution.fista import proximal_operator_lasso


def test_proximal_operator_lasso():
    a_all = np.array([1, -1, 2, -2, 0])
    a_thr = np.array([0, 0, 1, -1, 0])
    assert np.allclose(proximal_operator_lasso(a_all, 1), a_thr)
