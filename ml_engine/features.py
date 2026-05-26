import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes sales_staged.parquet data and adds ML features.
    """
    df = df.copy()
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # ── Time features ──────────────────────────────────────
    df['Year']        = df['Date'].dt.year
    df['Month']       = df['Date'].dt.month
    df['Week_of_Year']= df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']     = df['Date'].dt.quarter

    # ── Lag features (previous weeks sales per store) ──────
    df['Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
    df['Lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)

    # ── Rolling average (4 week window per store) ──────────
    df['Rolling_Mean_4'] = (
        df.groupby('Store')['Weekly_Sales']
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    # ── Drop rows where lag/rolling created NaN ────────────
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_parquet('data/staged/sales_staged.parquet')
    df_features = build_features(df)
    print(" Features built!")
    print(f"Shape: {df_features.shape}")
    print(df_features[['Store','Date','Weekly_Sales','Lag_1','Rolling_Mean_4']].head())
    df_features.to_parquet('data/curated/sales_analytics.parquet', index=False)
    print(" Saved to data/curated/sales_analytics.parquet")