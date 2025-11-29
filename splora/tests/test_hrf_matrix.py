"""Tests for HRF matrix generation using pySPFM."""

import numpy as np

from pySPFM.deconvolution.hrf_generator import HRFMatrix


def test_HRFMatrix_shape():
    """Test HRF matrix has correct shape."""
    n_scans = 200
    hrf_obj = HRFMatrix(te=[0])
    hrf_obj.generate_hrf(tr=2, n_scans=n_scans)
    assert hrf_obj.hrf_.shape == (n_scans, n_scans)


def test_HRFMatrix_multi_echo():
    """Test HRF matrix for multi-echo data."""
    n_scans = 100
    te = [14.0, 29.0, 44.0]
    hrf_obj = HRFMatrix(te=te)
    hrf_obj.generate_hrf(tr=2, n_scans=n_scans)
    # Multi-echo should stack the HRF matrices for each echo
    assert hrf_obj.hrf_.shape[0] == n_scans * len(te)
    assert hrf_obj.hrf_.shape[1] == n_scans


def test_HRFMatrix_block_model():
    """Test HRF matrix with block model."""
    n_scans = 100
    hrf_obj = HRFMatrix(te=[0], block=True)
    hrf_obj.generate_hrf(tr=2, n_scans=n_scans)
    assert hrf_obj.hrf_.shape == (n_scans, n_scans)
