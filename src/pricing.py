import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Standard Black-Scholes formula for a European call option.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Annual risk-free interest rate.
    sigma : float
        Implied volatility (decimal).

    Returns
    -------
    price : float
        The theoretical price of the European call option.
    """
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def local_vol_monte_carlo(S0, K, T, r, spline_model, q=0.0002, n_sims=10000, n_steps=50):
    """
    Prices a European call using Monte Carlo simulation under local volatility.
    
    Parameters
    ----------
    S0 : float : Initial stock price
    K : float : Strike price
    T : float : Time to maturity (years)
    r : float : Risk-free rate
    spline_model : RectBivariateSpline : The fitted IV spline to derive local vol
    q : float, optional : Continuous dividend yield (default is 0.0002)
    """
    dt = T / n_steps
    S_t = np.full(n_sims, S0)
    
    for i in range(n_steps):
        t_curr = i * dt
        sigma_local = spline_model(t_curr, S_t, grid=False) 
        Z = np.random.standard_normal(n_sims)
        S_t *= np.exp(((r - q) - 0.5 * sigma_local**2) * dt + sigma_local * np.sqrt(dt) * Z)
        
    payoff = np.maximum(S_t - K, 0)
    return np.exp(-r * T) * np.mean(payoff)
