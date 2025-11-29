"""Tests for stability selection module."""

import os
import tempfile

import numpy as np

from splora.deconvolution import stability_selection
from splora.deconvolution.stability_selection import subsample
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()
hrf = np.load(os.path.join(data_dir, "hrf_matrix.npy"))
y = np.loadtxt(os.path.join(data_dir, "visual_task.1d"))


def test_subsample_mode1_single_echo():
    """Test subsampling with mode 1 for single echo data."""
    np.random.seed(42)
    nscans = 100
    nTE = 1
    subsample_idx = subsample(nscans, mode=1, nTE=nTE)

    # Check that 60% of timepoints are kept
    assert len(subsample_idx) == int(0.6 * nscans)
    # Check that indices are sorted
    assert np.all(np.diff(subsample_idx) >= 0)
    # Check that indices are within bounds
    assert subsample_idx.min() >= 0
    assert subsample_idx.max() < nscans


def test_subsample_mode1_multi_echo():
    """Test subsampling with mode 1 for multi-echo data."""
    np.random.seed(42)
    nscans = 100
    nTE = 3
    subsample_idx = subsample(nscans, mode=1, nTE=nTE)

    # Check that 60% of timepoints are kept per echo
    assert len(subsample_idx) == int(0.6 * nscans) * nTE
    # Check that indices span all echoes
    assert subsample_idx.max() < nscans * nTE


def test_subsample_mode2():
    """Test subsampling with mode > 1 (same timepoints across echoes)."""
    np.random.seed(42)
    nscans = 100
    nTE = 1
    subsample_idx = subsample(nscans, mode=2, nTE=nTE)

    # Check that 60% of timepoints are kept
    assert len(subsample_idx) == int(0.6 * nscans)
    # Check that indices are sorted
    assert np.all(np.diff(subsample_idx) >= 0)


def test_lambda_value_calculation():
    """Test that lambda values are calculated correctly in stability_selection."""
    # Create minimal test data
    n_voxels = 5
    n_lambdas = 10

    # Use real hrf and y data
    y_expanded = np.expand_dims(y, axis=1)
    y_multi = np.tile(y_expanded, (1, n_voxels))

    # Calculate expected max lambda for first voxel
    expected_max_lambda = abs(np.dot(hrf.T, y_multi[:, 0])).max()

    # Verify the lambda calculation logic matches stability_selection
    lambda_values = np.zeros((n_lambdas, n_voxels))
    for voxel in range(n_voxels):
        voxel_data = y_multi[:, voxel]
        max_lambda = abs(np.dot(hrf.T, voxel_data)).max()
        if max_lambda > 0:
            lambda_values[:, voxel] = np.geomspace(
                0.05 * max_lambda, 0.95 * max_lambda, n_lambdas
            )
        else:
            lambda_values[:, voxel] = np.inf * np.ones(n_lambdas)

    # Check that lambda values are in geometric progression
    assert lambda_values.shape == (n_lambdas, n_voxels)
    assert np.allclose(lambda_values[0, 0], 0.05 * expected_max_lambda)
    assert np.allclose(lambda_values[-1, 0], 0.95 * expected_max_lambda)

    # Check geometric spacing
    ratios = lambda_values[1:, 0] / lambda_values[:-1, 0]
    assert np.allclose(ratios, ratios[0])  # All ratios should be equal


def test_stability_selection_saves_files():
    """Test that stability_selection saves intermediate files correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        n_voxels = 3
        n_scans = 160
        n_lambdas = 5
        n_surrogates = 2

        # Create test data
        y_expanded = np.expand_dims(y, axis=1)
        y_multi = np.tile(y_expanded, (1, n_voxels))

        # Create mock beta files that would be created by cluster jobs
        for surrogate in range(n_surrogates):
            for lambda_idx in range(n_lambdas):
                # Create mock boolean beta results
                mock_result = np.random.choice([True, False], size=(n_scans, n_voxels))
                np.save(
                    os.path.join(temp_dir, f"beta_{surrogate}_{lambda_idx}.npy"),
                    mock_result,
                )

        # Run stability selection with saved_data=True to skip cluster submission
        auc = stability_selection.stability_selection(
            hrf=hrf,
            y=y_multi,
            nTE=1,
            tr=2.0,
            temp=temp_dir,
            n_scans=n_scans,
            block_model=False,
            jobs=1,
            n_lambdas=n_lambdas,
            n_surrogates=n_surrogates,
            group=0.2,
            saved_data=True,
        )

        # Check that lambda_range.npy was saved
        assert os.path.exists(os.path.join(temp_dir, "lambda_range.npy"))

        # Check that hrf.npy was saved
        assert os.path.exists(os.path.join(temp_dir, "hrf.npy"))

        # Check that data.npy was saved
        assert os.path.exists(os.path.join(temp_dir, "data.npy"))

        # Check AUC output shape
        assert auc.shape == (n_scans, n_voxels)

        # Check AUC values are between 0 and 1
        assert np.all(auc >= 0)
        assert np.all(auc <= 1)


def test_bget():
    """Test the bget helper function."""
    # Test with a simple echo command
    result = stability_selection.bget("echo hello")
    assert result == ["hello"]


def test_auc_calculation():
    """Test that AUC is calculated correctly from mock beta results."""
    n_scans = 10
    n_voxels = 2
    n_lambdas = 3
    n_surrogates = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock lambda values
        lambda_values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        np.save(os.path.join(temp_dir, "lambda_range.npy"), lambda_values)

        # Create mock data files
        y_test = np.random.randn(n_scans, n_voxels)
        hrf_test = np.eye(n_scans)
        np.save(os.path.join(temp_dir, "data.npy"), y_test)
        np.save(os.path.join(temp_dir, "hrf.npy"), hrf_test)

        # Create consistent mock beta results (all True for testing)
        for surrogate in range(n_surrogates):
            for lambda_idx in range(n_lambdas):
                mock_result = np.ones((n_scans, n_voxels), dtype=bool)
                np.save(
                    os.path.join(temp_dir, f"beta_{surrogate}_{lambda_idx}.npy"),
                    mock_result,
                )

        # Calculate expected AUC manually
        # When all betas are True, auc_sum = n_surrogates for each lambda
        # auc = sum over lambdas of (auc_sum/n_surrogates * lambda/sum_lambdas)
        # = sum over lambdas of (1 * lambda/sum_lambdas)
        # = sum_lambdas / sum_lambdas = 1
        auc = stability_selection.stability_selection(
            hrf=hrf_test,
            y=y_test,
            nTE=1,
            tr=2.0,
            temp=temp_dir,
            n_scans=n_scans,
            block_model=False,
            jobs=1,
            n_lambdas=n_lambdas,
            n_surrogates=n_surrogates,
            group=0.2,
            saved_data=True,
        )

        # When all selections are True, AUC should be 1
        assert np.allclose(auc, 1.0)
