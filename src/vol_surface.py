import numpy as np
from scipy.interpolate import RectBivariateSpline

def build_volatility_surfaces(df, r=0.04, q=0.0002):
    """
    Fits a Bivariate Spline to market IV and derives the local volatility surface.

    This function uses the Dupire Equation to transform implied volatility
    into local volatility using analytical derivatives from the spline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned options data containing 'strike', 'T', 'S', and 'iv'.
    r : float, optional
        The risk-free interest rate (default is 0.04).
    q : float, optional
        Continuous dividend yield (default is 0.0002).

    Returns
    -------
    K_grid : np.ndarray
        2D grid of strike prices.
    T_grid : np.ndarray
        2D grid of maturities.
    iv_surface : np.ndarray
        2D grid of interpolated Implied Volatilities.
    local_vol : np.ndarray
        2D grid of calculated Local Volatilities (Dupire).
    spline : scipy.interpolate.RectBivariateSpline
        The fitted spline object for off-grid lookups.
    """
    # Setup Grid
    unique_T = np.sort(df["T"].unique())
    unique_K = np.sort(df["strike"].unique())
    
    # Pivot for spline
    pivot_iv = df.pivot(index='T', columns='strike', values='iv')
    pivot_iv = pivot_iv.interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)
    pivot_iv = pivot_iv.interpolate(method='linear', axis=0).ffill(axis=0).bfill(axis=0)

    # Fit RectBivariateSpline
    spline = RectBivariateSpline(unique_T, unique_K, pivot_iv.values, kx=3, ky=3, s=0.15)

    # Generate dense grid
    K_vals = np.linspace(df["strike"].min(), df["strike"].max(), 80)
    T_vals = np.linspace(df["T"].min(), df["T"].max(), 80)
    K_grid, T_grid = np.meshgrid(K_vals, T_vals)

    # Analytical derivatives
    iv_surface = spline(T_vals, K_vals)
    dV_dT = spline(T_vals, K_vals, dx=1, dy=0)
    dV_dK = spline(T_vals, K_vals, dx=0, dy=1)
    d2V_dK2 = spline(T_vals, K_vals, dx=0, dy=2)

    # Dupire formula
    S_now = df['S'].iloc[0]
    d1 = (np.log(S_now / K_grid) + ((r - q) + 0.5 * iv_surface**2) * T_grid) / (iv_surface * np.sqrt(T_grid))

    numerator = iv_surface**2 + 2 * iv_surface * T_grid * (dV_dT + (r - q) * K_grid * dV_dK)
    denominator = (1 + K_grid * d1 * np.sqrt(T_grid) * dV_dK)**2 + \
                  (iv_surface * K_grid**2 * T_grid) * (d2V_dK2 - d1 * np.sqrt(T_grid) * (dV_dK**2))

    local_vol = np.sqrt(np.clip(numerator / np.maximum(denominator, 1e-4), 0.01, 2.5))

    return K_grid, T_grid, iv_surface, local_vol, spline
