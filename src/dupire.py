import numpy as np
from scipy.interpolate import griddata
from bs import bs_call_price


def build_vol_surface(strikes, maturities, vols, K_grid, T_grid):
    """
    Interpolate implied volatility surface.

    Parameters:
    strikes : array
    maturities : array
    vols : array
    K_grid : meshgrid for strikes
    T_grid : meshgrid for maturities

    Returns:
    vol_surface (2D array)
    """
    points = np.column_stack((strikes, maturities))
    values = vols

    vol_surface = griddata(points, values, (K_grid, T_grid), method='cubic')

    return vol_surface


def compute_local_vol_surface(price_surface, K_grid, T_grid):
    """
    Compute local volatility surface using Dupire formula.
    """

    # Compute derivatives
    dC_dT = np.gradient(price_surface, T_grid[:, 0], axis=0)

    dC_dK = np.gradient(price_surface, K_grid[0, :], axis=1)
    d2C_dK2 = np.gradient(dC_dK, K_grid[0, :], axis=1)

    # Dupire formula
    denominator = 0.5 * K_grid**2 * d2C_dK2

    # Avoid division issues
    denominator = np.where(np.abs(denominator) < 1e-8, np.nan, denominator)

    local_var = dC_dT / denominator

    # Remove negative values (numerical artifacts)
    local_var = np.maximum(local_var, 0)

    return np.sqrt(local_var)
