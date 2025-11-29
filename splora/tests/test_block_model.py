
import numpy as np
from pySPFM.deconvolution.hrf_generator import HRFMatrix
from splora.deconvolution import fista

def test_block_model_delay():
    """Test that block model estimation does not introduce delays."""
    tr = 2.0
    n_scans = 100
    te = [0]
    
    # Generate HRF matrices
    hrf_obj_block = HRFMatrix(te=te, block=True)
    hrf_block = hrf_obj_block.generate_hrf(tr=tr, n_scans=n_scans).hrf_
    
    # Generate ground truth block signal
    # Innovation signal: +1 at start of block, -1 at end
    n_voxels = 1
    s_innovation = np.zeros((n_scans, n_voxels))
    true_onsets = [20, 40, 60, 80]
    s_innovation[20, :] = 1
    s_innovation[40, :] = -1
    s_innovation[60, :] = 1
    s_innovation[80, :] = -1
    
    # BOLD signal
    y = np.dot(hrf_block, s_innovation)
    
    # Add some noise
    np.random.seed(42)
    y_noisy = y + 0.01 * np.random.randn(*y.shape)
    
    # Run FISTA
    S_est, _, _, _, _, L_est = fista.fista(
        hrf=hrf_block,
        y=y_noisy,
        n_te=1,
        block_model=True,
        pfm_only=True, # Use PFM only to force S to capture signal
        max_iter=50,
        lambda_crit="factor",
        factor=0.1, # Low regularization to ensure we catch the signal
        tr=tr,
        te=te,
        eigen_thr=0.001 # Low threshold
    )
    
    # Check onsets
    est_onsets = np.where(np.abs(S_est) > 0.1)[0]
    
    # We expect the estimated onsets to be close to the true onsets
    # Allow for +/- 1 TR error, although with this synthetic data it should be exact
    matched_onsets = 0
    for true_onset in true_onsets:
        if np.any(np.abs(est_onsets - true_onset) <= 1):
            matched_onsets += 1
            
    assert matched_onsets == len(true_onsets), f"Failed to recover all onsets. True: {true_onsets}, Est: {est_onsets}"
    
    # Check that we don't have too many false positives (dense signal)
    # The bug caused the signal to be dense (block-like) instead of sparse (innovation-like)
    # With noise, we might get some smearing, but it shouldn't be a full block (approx 40 points)
    assert len(est_onsets) < 30, f"Estimated signal is too dense, likely block-like instead of innovation-like. Non-zeros: {len(est_onsets)}"

    # Check reconstruction MSE
    y_fitted = np.dot(hrf_block, S_est)
    residual = y_noisy - y_fitted - L_est
    mse = np.mean(residual**2)
    
    # MSE should be comparable to noise level
    # Note: Regularization introduces bias, so MSE will be higher than noise variance (0.0001)
    assert mse < 0.02, f"MSE is too high: {mse}"
