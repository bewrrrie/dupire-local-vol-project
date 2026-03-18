import pandas as pd

def load_and_clean_data(file_path):
    """
    Loads raw NVDA options data, cleans headers, and filters for a specific snapshot.

    Parameters
    ----------
    file_path : str
        The relative or absolute path to the CSV file containing options data.

    Returns
    -------
    df : pd.DataFrame
        A cleaned DataFrame sorted by maturity (T) and strike, containing 
        mean IV and mid prices for the most recent quote date.
    snapshot_date : datetime
        The specific date extracted for the volatility surface snapshot.
    """
    df_raw = pd.read_csv(file_path, low_memory=False)
    
    # Extract and clean columns
    df = df_raw[[
        " [QUOTE_DATE]", " [EXPIRE_DATE]", " [STRIKE]", " [UNDERLYING_LAST]",
        " [C_BID]", " [C_ASK]", " [C_VOLUME]", " [C_IV]"
    ]].copy()
    df.columns = ["date", "expire", "strike", "S", "bid", "ask", "volume", "iv"]

    # Convert Types
    df["date"] = pd.to_datetime(df["date"])
    df["expire"] = pd.to_datetime(df["expire"])
    cols_to_fix = ["bid", "ask", "volume", "strike", "S", "iv"]
    df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

    # Filters
    df = df[df['iv'] > 0].dropna()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["volume"] > 5)]
    df = df[(df['ask'] - df['bid']) / df['mid'] < 0.25]
    
    df["T"] = (df["expire"] - df["date"]).dt.days / 365
    df = df[(df["T"] > 0.02) & (df["T"] < 1.0)]
    
    df['moneyness'] = df['strike'] / df['S']
    df = df[(df['moneyness'] > 0.6) & (df['moneyness'] < 1.4)]

    # Snapshot & grouping
    snapshot_date = df['date'].max()
    df = df[df['date'] == snapshot_date].copy()
    df = df.groupby(['strike', 'expire', 'T', 'S']).agg({'iv': 'mean', 'mid': 'mean'}).reset_index()
    
    return df.sort_values(['T', 'strike']), snapshot_date
