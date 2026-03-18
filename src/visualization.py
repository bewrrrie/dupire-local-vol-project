import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import seaborn as sns


def plot_vol_smile(df, target_T, figures_dir="."):
    """
    Plots the implied volatility ''smile'' for a specific maturity.

    This visualization compares the actual market-implied volatility across 
    different strike prices against the constant volatility assumption 
    used in the standard Black-Scholes model. It identifies the 'skew' 
    or 'smirk' present in equity markets.

    Args:
        df (pd.DataFrame): Dataframe containing 'strike', 'iv' (implied volatility), 
            and 'T' (time to maturity) columns.
        target_T (float): The desired time to maturity (in years) to visualize. 
            The function will select the market expiry closest to this value.

    Returns:
        None: Displays a Matplotlib plot showing the volatility curve.
    """
    available_Ts = df['T'].unique()
    closest_T = available_Ts[np.abs(available_Ts - target_T).argmin()]
    smile_data = df[df['T'] == closest_T]

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=smile_data, x='strike', y='iv', marker='o', label=f'Market IV (T={closest_T:.2f})')
    
    atm_vol = smile_data.iloc[len(smile_data)//2]['iv']
    plt.axhline(y=atm_vol, color='r', linestyle='--', label='BS Assumption (Flat Vol)')

    plt.title(f"NVDA Volatility Smile: Market Reality vs. BS Assumption")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{figures_dir}/nvda_vol_smile.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_volatility_surfaces(K_grid, T_grid, iv_surf, local_vol, snapshot_date, figures_dir="."):
    """
    Generates 3D (static) visualization of implied and
    local volatility surfaces.

    Parameters
    ----------
    K_grid : np.ndarray
        2D grid of strike prices.
    T_grid : np.ndarray
        2D grid of maturities.
    iv_surf : np.ndarray
        2D grid of Implied Volatility values.
    local_vol : np.ndarray
        2D grid of Local Volatility values.
    snapshot_date : datetime
        The date of the market snapshot for the title.
    """
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(K_grid, T_grid, iv_surf, cmap=cm.viridis, antialiased=True)
    ax1.set_title(f"Implied Volatility Surface (Market)\n{snapshot_date.date()}")
    ax1.set_xlabel("Strike price, K"); ax1.set_ylabel("Time to maturity, T"); ax1.set_zlabel("IV")
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(K_grid, T_grid, local_vol, cmap=cm.plasma, antialiased=True)
    ax2.set_title("Local Volatility Surface (Dupire Stabilized)")
    ax2.set_xlabel("Strike price, K"); ax2.set_ylabel("Time to maturity, T"); ax2.set_zlabel("Local Vol")
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/nvda_vol_surf.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_volatility_surfaces_plotly(K_grid, T_grid, iv_surf, local_vol, snapshot_date):
    """
    Generates 3D interactive visualization of implied and
    local volatility surfaces using Plotly.

    Parameters
    ----------
    K_grid : np.ndarray
        2D grid of strike prices.
    T_grid : np.ndarray
        2D grid of maturities.
    iv_surf : np.ndarray
        2D grid of Implied Volatility values.
    local_vol : np.ndarray
        2D grid of Local Volatility values.
    snapshot_date : datetime
        The date of the market snapshot for the title.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=(
            f"Implied Volatility (Market) - {snapshot_date.date()}", 
            "Local Volatility (Dupire Stabilized)"
        ),
        horizontal_spacing=0.05
    )

    fig.add_trace(
        go.Surface(
            x=K_grid, y=T_grid, z=iv_surf, 
            colorscale='Viridis', 
            name='Implied Vol',
            colorbar=dict(title='IV', x=-0.07)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Surface(
            x=K_grid, y=T_grid, z=local_vol, 
            colorscale='Plasma', 
            name='Local Vol',
            colorbar=dict(title='Local Vol', x=1.02)
        ),
        row=1, col=2
    )

    fig.update_layout(
        title={
            'text': "Volatility Surface Comparison",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        width=1100,
        height=600,
        margin=dict(l=50, r=50, b=50, t=100),
        scene=dict(
            xaxis_title="Strike price, K",
            yaxis_title="Time to maturity, T",
            zaxis_title="Implied Volatility"
        ),
        scene2=dict(
            xaxis_title="Strike price, K",
            yaxis_title="Time to maturity, T",
            zaxis_title="Local Volatility"
        )
    )

    fig.show()
