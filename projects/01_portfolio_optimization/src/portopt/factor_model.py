"""Rolling FF3 factor model (extracted verbatim, ROADMAP 2.2)."""
import numpy as np
import statsmodels.api as sm


def estimate_factor_model(returns, ff_factors, date, lookback=252):
    """
    Estimate FF3 factor model for all assets using trailing window.
    
    Returns:
        alphas: (n_assets,) array of alpha estimates
        betas:  (n_assets, 3) array of factor betas
        mu_hat: (n_assets,) array of expected returns
        residual_vol: (n_assets,) array of idiosyncratic volatility
    """
    # Get trailing window
    end_idx = returns.index.get_loc(date)
    start_idx = max(0, end_idx - lookback)
    
    ret_window = returns.iloc[start_idx:end_idx]
    ff_window = ff_factors.loc[ret_window.index]
    
    n = ret_window.shape[1]
    alphas = np.zeros(n)
    betas = np.zeros((n, 3))
    residual_vol = np.zeros(n)
    
    # Factor matrix: MKT-RF, SMB, HML
    X = ff_window[['Mkt-RF', 'SMB', 'HML']].values
    X_with_const = sm.add_constant(X)
    
    # Excess returns
    rf = ff_window['RF'].values
    
    for i in range(n):
        y = ret_window.iloc[:, i].values - rf
        
        try:
            model = sm.OLS(y, X_with_const).fit()
            alphas[i] = model.params[0]
            betas[i, :] = model.params[1:]
            residual_vol[i] = model.resid.std()
        except Exception:
            alphas[i] = 0.0
            betas[i, :] = [1.0, 0.0, 0.0]
            residual_vol[i] = y.std() if len(y) > 0 else 0.01
    
    # Expected returns: alpha + beta * E[factors]
    factor_means = ff_window[['Mkt-RF', 'SMB', 'HML']].mean().values
    mu_hat = alphas + betas @ factor_means
    
    return alphas, betas, mu_hat, residual_vol
