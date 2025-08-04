# src/features/generate_features.py
def create_time_features(df):
    # Lag features
    df['power_lag_1h'] = df['power_kw'].shift(4)  # 4x15min=1h

    # Rolling statistics
    df['power_ma_6h'] = df['power_kw'].rolling(24).mean()

    # Event indicators
    df['is_working_hours'] = df['hour'].between(8, 18).astype(int)
    return df