"""Covariance estimators: sample, Ledoit-Wolf, EWMA (extracted verbatim)."""
import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns_window, method='ledoit_wolf', ewma_halflife=63):
    """
    Estimate covariance matrix using specified method.
    
    Args:
        returns_window: (T, n) DataFrame of returns
        method: 'sample', 'ledoit_wolf', or 'ewma'
        ewma_halflife: halflife for EWMA in trading days
        
    Returns:
        cov_matrix: (n, n) numpy array, guaranteed PSD
        metadata: dict with estimation details
    """
    X = returns_window.values
    n = X.shape[1]
    meta = {'method': method, 'n_obs': X.shape[0], 'n_assets': n}
    
    if method == 'sample':
        cov = np.cov(X, rowvar=False)
        meta['condition_number'] = np.linalg.cond(cov)
        
    elif method == 'ledoit_wolf':
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
        meta['shrinkage_intensity'] = lw.shrinkage_
        meta['condition_number'] = np.linalg.cond(cov)
        
    elif method == 'ewma':
        # Exponentially weighted covariance
        alpha = 1 - np.exp(-np.log(2) / ewma_halflife)
        T = X.shape[0]
        weights = np.array([(1 - alpha) ** i for i in range(T - 1, -1, -1)])
        weights /= weights.sum()
        
        X_demean = X - np.average(X, weights=weights, axis=0)
        cov = (X_demean * weights[:, np.newaxis]).T @ X_demean
        meta['ewma_halflife'] = ewma_halflife
        meta['effective_obs'] = 1 / (weights ** 2).sum()
        meta['condition_number'] = np.linalg.cond(cov)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure PSD via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov)
    min_eigval = eigvals.min()
    if min_eigval < 0:
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        meta['psd_clipped'] = True
        meta['min_eigval_before'] = min_eigval
    else:
        meta['psd_clipped'] = False
    
    return cov, meta
