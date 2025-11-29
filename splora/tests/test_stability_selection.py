"""Tests for stability selection module."""

import os

import numpy as np
import pytest

from splora.deconvolution import stability_selection
from splora.deconvolution.stability_selection import (
    calculate_auc,
    calculate_lambda_range,
    get_subsampling_indices,
)
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(os.path.join(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(os.path.join(data_dir, "visual_task.1d"))


def test_get_subsampling_indices_single_echo():
    """Test subsampling indices for single echo data."""
    np.random.seed(42)
    n_scans = 100
    n_te = 1
    subsample_idx = get_subsampling_indices(n_scans, n_te=n_te)

    # Check that 60% of timepoints are kept
    assert len(subsample_idx) == int(0.6 * n_scans)
    # Check that indices are sorted
    assert np.all(np.diff(subsample_idx) >= 0)
    # Check that indices are within bounds
    assert subsample_idx.min() >= 0
    assert subsample_idx.max() < n_scans


def test_get_subsampling_indices_multi_echo():
    """Test subsampling indices for multi-echo data."""
    np.random.seed(42)
    n_scans = 100
    n_te = 3
    subsample_idx = get_subsampling_indices(n_scans, n_te=n_te)

    # Check that 60% of timepoints are kept per echo, same timepoints across echoes
    assert len(subsample_idx) == int(0.6 * n_scans) * n_te
    # Check that indices span all echoes
    assert subsample_idx.max() < n_scans * n_te


def test_get_subsampling_indices_custom_ratio():
    """Test subsampling with custom ratio."""
    np.random.seed(42)
    n_scans = 100
    n_te = 1
    ratio = 0.5
    subsample_idx = get_subsampling_indices(n_scans, n_te=n_te, ratio=ratio)

    # Check that 50% of timepoints are kept
    assert len(subsample_idx) == int(ratio * n_scans)


def test_calculate_lambda_range():
    """Test that lambda values are calculated correctly."""
    n_voxels = 5
    n_lambdas = 10

    # Use real hrf and y data
    y_expanded = np.expand_dims(y, axis=1)
    y_multi = np.tile(y_expanded, (1, n_voxels))

    # Calculate expected max lambda for first voxel
    expected_max_lambda = abs(np.dot(hrf.T, y_multi[:, 0])).max()

    # Calculate lambda range
    lambda_values = calculate_lambda_range(hrf, y_multi, n_lambdas)

    # Check shape
    assert lambda_values.shape == (n_lambdas, n_voxels)

    # Check bounds
    assert np.allclose(lambda_values[0, 0], 0.05 * expected_max_lambda)
    assert np.allclose(lambda_values[-1, 0], 0.95 * expected_max_lambda)

    # Check geometric spacing
    ratios = lambda_values[1:, 0] / lambda_values[:-1, 0]
    assert np.allclose(ratios, ratios[0])  # All ratios should be equal


def test_calculate_lambda_range_zero_data():
    """Test lambda calculation with zero data returns inf."""
    n_voxels = 2
    n_lambdas = 5
    n_scans = 10

    # Create data where one voxel is all zeros
    y_test = np.random.randn(n_scans, n_voxels)
    y_test[:, 1] = 0  # Zero out second voxel
    hrf_test = np.random.randn(n_scans, n_scans)

    lambda_values = calculate_lambda_range(hrf_test, y_test, n_lambdas)

    # First voxel should have finite values
    assert np.all(np.isfinite(lambda_values[:, 0]))
    # Second voxel should have inf values
    assert np.all(np.isinf(lambda_values[:, 1]))


def test_calculate_auc():
    """Test AUC calculation from surrogate results."""
    n_lambdas = 3
    n_scans = 10
    n_voxels = 2
    n_surrogates = 2

    # Create mock results from surrogates (all ones = always selected)
    results = np.ones((n_lambdas, n_scans, n_voxels), dtype=np.int8)
    lambda_values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    # Create list of (results, lambda_values) tuples
    all_results = [(results, lambda_values) for _ in range(n_surrogates)]

    auc = calculate_auc(all_results, n_surrogates)

    # When all selections are 1, AUC should be 1
    assert auc.shape == (n_scans, n_voxels)
    assert np.allclose(auc, 1.0)


def test_calculate_auc_zeros():
    """Test AUC calculation with zero selection frequencies."""
    n_lambdas = 3
    n_scans = 10
    n_voxels = 2
    n_surrogates = 2

    # Create mock results from surrogates (all zeros = never selected)
    results = np.zeros((n_lambdas, n_scans, n_voxels), dtype=np.int8)
    lambda_values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    # Create list of (results, lambda_values) tuples
    all_results = [(results, lambda_values) for _ in range(n_surrogates)]

    auc = calculate_auc(all_results, n_surrogates)

    # When all selections are 0, AUC should be 0
    assert np.allclose(auc, 0.0)


def test_calculate_auc_partial():
    """Test AUC calculation with partial selection frequencies."""
    n_lambdas = 3
    n_scans = 10
    n_voxels = 1
    n_surrogates = 2

    # Create selection frequencies where only first lambda selects
    results = np.zeros((n_lambdas, n_scans, n_voxels), dtype=np.int8)
    results[0, :, :] = 1

    # Create lambda values (1, 2, 3) with sum = 6
    lambda_values = np.array([[1.0], [2.0], [3.0]])

    # Create list of (results, lambda_values) tuples
    all_results = [(results, lambda_values) for _ in range(n_surrogates)]

    auc = calculate_auc(all_results, n_surrogates)

    # AUC should be 1 * (1/6) = 1/6
    assert np.allclose(auc, 1.0 / 6.0)


@pytest.mark.slow
def test_stability_selection_runs():
    """Test that stability selection runs without errors (minimal test)."""
    n_voxels = 2
    n_scans = 160

    # Use a small number of surrogates and lambdas for speed
    n_lambdas = 3
    n_surrogates = 2

    # Create test data
    y_expanded = np.expand_dims(y, axis=1)
    y_multi = np.tile(y_expanded, (1, n_voxels))

    # Run stability selection
    auc = stability_selection.stability_selection(
        hrf=hrf,
        y=y_multi,
        n_te=1,
        tr=2.0,
        n_scans=n_scans,
        block_model=False,
        n_jobs=2,
        n_lambdas=n_lambdas,
        n_surrogates=n_surrogates,
        group=0.2,
    )

    # Check output shape
    assert auc.shape == (n_scans, n_voxels)

    # Check AUC values are non-negative
    assert np.all(auc >= 0)
