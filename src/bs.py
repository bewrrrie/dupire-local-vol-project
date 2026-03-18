import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call_price(S, K, T, r, sigma):
    """
    Black-Scholes price of a European call option.

    Parameters:
    S : float or array - underlying price
    K : float or array - strike
    T : float or array - time to maturity
    r : float - risk-free rate
    sigma : float or array - volatility

    Returns:
    Call price
    """
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(C, S, K, T, r=0.01):
    """
    Compute implied volatility using Brent's method.

    Parameters:
    C : float - market price
    S : float - underlying price
    K : float - strike
    T : float - time to maturity
    r : float - risk-free rate

    Returns:
    Implied volatility (float) or np.nan if failed
    """
    if C <= 0 or T <= 0:
        return np.nan

    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - C

    try:
        return brentq(objective, 1e-6, 5.0)
    except:
        return np.nan
