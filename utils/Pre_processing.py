# Imports
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, OAS


def compute_mu_sigma(prices_df, lookback=252, shrinkage=True):
        """
        Takes dataframe of prices and computes ROI and Covariance matrix
    
        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame of price data with index as dates and columns as asset IDs
        lookback : int (default = 252)
            Total number of trading days (USA 252 trading days in a year).
        shrinkage : boolean (default = True)
            Shrinks sample covariance towards a more stable target using LedoitWold shrinkage estimator.
    
        Returns
        ----------
        mu : np.array
            Array of ROI of each asset
        Sigma : np.ndarray
            n x n covariance matrix where n is the number of assets
        """
        # compute daily returns
        rets = prices_df.pct_change().dropna() # percentage change between the current element and prior element
        mu_daily = rets.mean()
        mu_annual = mu_daily * lookback

        # Shrinks sample covariance towards a more stable target
        if shrinkage:
            # checks whether to use OAS or LedoitWolf
            n_assets = prices_df.shape[1]  
            n_periods = prices_df.shape[0]  
            ratio = n_assets / n_periods
            
            if ratio > 0.2: # High dimensionality
                lw = LedoitWolf().fit(rets.values)
                Sigma_daily = lw.covariance_
            else:
                oas = OAS().fit(rets.values) # more aggressive shrinking
                Sigma_daily = oas.covariance_
        else: # No shrinking
            Sigma_daily = np.cov(rets.values.T)
            
        Sigma_annual = Sigma_daily * lookback
        mu = mu_annual.values
        Sigma = Sigma_annual
            
        return mu, Sigma