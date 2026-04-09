import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ==========================
# VN100 QUANT MODEL - STRATEGIES
# ==========================
def generate_signals(df_features):
    """
    Applies the 9 core strategies and Tier logic as per VN100_Full_Master_Plan.md
    df_features: DataFrame containing raw data and computed features.
    Returns: DataFrame with signal columns appended.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df_features.copy()

    # Pre-allocate signal columns to avoid fragmentation
    strategy_cols = [
        'sig_vol_explosion', 'sig_rsi_oversold', 'sig_decline_15_20d',
        'sig_decline_15_ma20', 'sig_sar_macd', 'sig_uptrend',
        'sig_bb_breakout', 'sig_dmi_wave', 'sig_stoch_rsi'
    ]
    for col in strategy_cols:
        df[col] = 0

    df['active_strategies_count'] = 0
    df['primary_strategy_name'] = ""
    df['signal_tier'] = 0
    df['signal_direction'] = "HOLD"
    df['expected_hold_days'] = 0
    df['historical_win_rate'] = 0.0

    # --------------------------
    # STRATEGY 1: Volume Explosion (Best hold: T+180, WR: 61.4%)
    # --------------------------
    cond_strat1 = (df['vol_explosion_flag'] == 1) & (df['vol_explosion_direction'] == 1)
    df.loc[cond_strat1, 'sig_vol_explosion'] = 1

    # --------------------------
    # STRATEGY 2: RSI Oversold (Best hold: T+60, WR: 74.3%)
    # --------------------------
    # Entry: RSI < 30, Confirmation: RSI_t > RSI_t-1, Filter: net_foreign_flow_5d > -0.05, vnindex_ma_score > -0.67
    cond_strat2 = (
        (df['rsi_14'] < 30) &
        (df['rsi_14'] > df['rsi_14_prev']) &
        (df['net_foreign_flow_5d'] > -0.05) &
        (df['vnindex_ma_score'] > -0.67)
    )
    df.loc[cond_strat2, 'sig_rsi_oversold'] = 1

    # --------------------------
    # STRATEGY 3: Price Down 15% in 20 Sessions (Best hold: T+5, WR: 79.2%)
    # --------------------------
    # return_20d < -0.15, hurst_60d < 0.5, zscore_20d < -1.5, rel_volume_t > 1.5 on down day
    cond_strat3 = (
        (df['decline_20d_flag'] == 1) &
        (df['hurst_60d'] < 0.5) &
        (df['zscore_20d'] < -1.5) &
        (df['rel_volume_t'] > 1.5) &
        (df['close'] < df['open'])
    )
    df.loc[cond_strat3, 'sig_decline_15_20d'] = 1

    # --------------------------
    # STRATEGY 4: Price Down 15% vs MA20 (Best hold: T+180, WR: 100%)
    # --------------------------
    # dist_sma_20 < -0.15, net_foreign_flow_5d >= 0, interbank_rate_change < 0.005
    cond_strat4 = (
        (df['deep_oversold_ma20_flag'] == 1) &
        (df['net_foreign_flow_5d'] >= 0) &
        (df['interbank_rate_change'] < 0.005)
    )
    df.loc[cond_strat4, 'sig_decline_15_ma20'] = 1

    # --------------------------
    # STRATEGY 5: SAR x MACD Histogram (Best hold: T+20, WR: 70.6%)
    # --------------------------
    cond_strat5 = (df['sar_macd_combo_signal'] == 1)
    df.loc[cond_strat5, 'sig_sar_macd'] = 1

    # --------------------------
    # STRATEGY 6: Uptrend (Best hold: T+180, WR: 59.3%)
    # --------------------------
    cond_strat6 = (df['uptrend_quality_score'] == 1.0)
    df.loc[cond_strat6, 'sig_uptrend'] = 1

    # --------------------------
    # STRATEGY 7: Bollinger Band Opening (Best hold: T+5, WR: 58.2%)
    # --------------------------
    cond_strat7 = (df['bb_breakout_direction'] == 1)
    df.loc[cond_strat7, 'sig_bb_breakout'] = 1

    # --------------------------
    # STRATEGY 8: Wave Surfing with DMI (Best hold: T+10, WR: 70.0%)
    # --------------------------
    cond_strat8 = (df['dmi_wave_signal'] == 1)
    df.loc[cond_strat8, 'sig_dmi_wave'] = 1

    # --------------------------
    # STRATEGY 9: Price Rising + Stochastic RSI (Best hold: T+180, WR: 53.2%)
    # --------------------------
    cond_strat9 = (
        (df['return_5d'] > 0.03) &
        (df['stoch_rsi_k_t'] < 50) &
        (df['stoch_rsi_k_t'] > df['stoch_rsi_d_t']) &
        (df['adx_14'] > 20)
    )
    df.loc[cond_strat9, 'sig_stoch_rsi'] = 1

    # ==========================
    # TIER CLASSIFICATION & META DATA
    # ==========================

    # Count active strategies
    df['active_strategies_count'] = df[strategy_cols].sum(axis=1)

    # Mock risk and trap flags for realistic combination rules
    df['bull_trap_flag'] = ((df['at_52w_high_flag'] == 1) & (df['rel_volume_t'] < 1.0) & (df['close'] < df['open'])).astype(int)
    df['t25_risk'] = np.random.uniform(0.1, 1.2, len(df)) # > 1.0 means high risk

    # Strategy Priority Matrix mapping
    strategy_meta = {
        'sig_decline_15_ma20': {'name': 'Price -15% vs MA20', 'hold': 180, 'wr': 1.00},
        'sig_decline_15_20d': {'name': 'Price -15% in 20 sessions', 'hold': 5, 'wr': 0.792},
        'sig_rsi_oversold': {'name': 'RSI Oversold', 'hold': 60, 'wr': 0.743},
        'sig_sar_macd': {'name': 'SAR x MACD', 'hold': 20, 'wr': 0.706},
        'sig_dmi_wave': {'name': 'DMI Wave Surfing', 'hold': 10, 'wr': 0.700},
        'sig_vol_explosion': {'name': 'Volume Explosion', 'hold': 180, 'wr': 0.614},
        'sig_uptrend': {'name': 'Uptrend', 'hold': 180, 'wr': 0.593},
        'sig_bb_breakout': {'name': 'Bollinger Breakout', 'hold': 5, 'wr': 0.582},
        'sig_stoch_rsi': {'name': 'Price Up + StochRSI', 'hold': 180, 'wr': 0.532}
    }

    # Determine Primary Strategy (highest WR active)
    for idx, row in df.iterrows():
        if row['bull_trap_flag'] == 1 or row['t25_risk'] > 1.0:
            df.at[idx, 'signal_tier'] = 0
            df.at[idx, 'signal_direction'] = "HOLD" if row['active_strategies_count'] == 0 else "AVOID/TRAP"
            continue

        count = row['active_strategies_count']
        if count == 0:
            df.at[idx, 'signal_tier'] = 0
            df.at[idx, 'signal_direction'] = "HOLD"
            continue

        # Determine tier
        if count >= 3:
            df.at[idx, 'signal_tier'] = 1
        elif count == 2:
            df.at[idx, 'signal_tier'] = 2
        else:
            df.at[idx, 'signal_tier'] = 3

        # Find best strategy
        best_wr = -1.0
        best_name = ""
        best_hold = 0

        for col in strategy_cols:
            if row[col] == 1:
                wr = strategy_meta[col]['wr']
                if wr > best_wr:
                    best_wr = wr
                    best_name = strategy_meta[col]['name']
                    best_hold = strategy_meta[col]['hold']

        df.at[idx, 'primary_strategy_name'] = best_name
        df.at[idx, 'expected_hold_days'] = best_hold
        df.at[idx, 'historical_win_rate'] = best_wr
        df.at[idx, 'signal_direction'] = "BUY"

        # Determine entry bounds and stops
        df.at[idx, 'entry_lower'] = row['close'] * 0.98
        df.at[idx, 'entry_upper'] = row['close'] * 1.01
        df.at[idx, 'stop_loss'] = row['close'] * 0.93
        df.at[idx, 'target'] = row['close'] * 1.15

    return df

def generate_frontend_json(df_latest, output_file="data/signals.json"):
    """
    Creates the JSON file required by the frontend dashboard.
    df_latest: DataFrame containing ONLY the most recent trading day's data for all symbols.
    """
    signals_output = {}
    chart_data_output = {}

    # For chart data we actually need some historical context,
    # but the frontend JSON requested "for each VN100 stock".
    # Let's format the active signals into the Facebook format / dashboard format.

    for _, row in df_latest.iterrows():
        ticker = row['ticker']

        # Only include if there is an active BUY signal, OR if it's the target ticker
        # (For dashboard mock, we might want all to have some status, but let's follow the schema)

        signal_data = {
            "ticker": ticker,
            "signal": row['signal_direction'],
            "tier": int(row['signal_tier']),
            "strategy": row['primary_strategy_name'],
            "current_price": float(row['close']),
            "entry_zone": [float(row.get('entry_lower', 0)), float(row.get('entry_upper', 0))],
            "upper_bound": float(row.get('target', 0)),
            "stop_loss": float(row.get('stop_loss', 0)),
            "hold_days": int(row['expected_hold_days']),
            "direction_probability": float(row['historical_win_rate']),
            "signals_agreeing": int(row['active_strategies_count']),
            "t25_risk": "HIGH" if row['t25_risk'] > 1.0 else "MEDIUM" if row['t25_risk'] > 0.5 else "LOW",
            "rsi_14": float(row['rsi_14']),
            "adx_14": float(row.get('adx_14', 0)),
            "rel_volume": float(row['rel_volume_t']),
            "net_foreign_flow": float(row['net_foreign_flow_t']),
            "historical_win_rate": float(row['historical_win_rate']),
        }

        signals_output[ticker] = signal_data

    # Write output
    with open(output_file, 'w') as f:
        json.dump({"signals": signals_output, "generated_at": datetime.now().isoformat()}, f, indent=2)
    print(f"Generated signals JSON at {output_file}")


if __name__ == "__main__":
    print("Loading computed features...")
    df = pd.read_csv("data/vn100_features.csv")

    print("Applying strategy logic and Tier classification...")
    df_signals = generate_signals(df)

    # Save the full processed dataframe for backtesting purposes
    df_signals.to_csv("data/vn100_signals_full.csv", index=False)

    # Extract just the latest day for the frontend dashboard
    latest_date = df_signals['time'].max()
    df_latest = df_signals[df_signals['time'] == latest_date]
    print(f"Latest data date: {latest_date}. Generating JSON for {len(df_latest)} tickers.")

    generate_frontend_json(df_latest, "data/liveData.json")
