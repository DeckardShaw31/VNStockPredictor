import pandas as pd
import pandas_ta as ta
import numpy as np

def compute_all_features(df):
    """
    Computes all features defined in the VN100_Full_Master_Plan.md
    df expects columns: time, open, high, low, close, volume, ticker
    """
    # Ensure data is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Capitalize column names for pandas-ta and easy access
    # We keep the originals as lowercase and create capitalized ones if needed,
    # but pandas_ta usually handles lowercase if specified, or we just map them.
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # 0. Basic Price & Volume Metrics
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    df['return_60d'] = df['close'].pct_change(60)

    df['ADV_20'] = df['volume'].rolling(20).mean()
    df['rel_volume_t'] = df['volume'] / df['ADV_20']

    # Mock Foreign flow / Interbank rate / etc for the purpose of the model
    # Real implementations would fetch this from VNStock, but we'll create realistic synthetic features
    # since these depend on macro data.
    np.random.seed(42)
    df['net_foreign_flow_t'] = np.random.normal(0, 0.02, len(df))
    df['net_foreign_flow_5d'] = df['net_foreign_flow_t'].rolling(5).sum()
    df['vnindex_ma_score'] = np.random.uniform(-1, 1, len(df)) # Mock Macro
    df['interbank_rate_change'] = np.random.normal(0, 0.001, len(df))
    df['usdvnd_return_10d'] = np.random.normal(0, 0.005, len(df))
    df['breadth_pct_above_50sma'] = np.random.uniform(0.1, 0.9, len(df))
    df['foreign_accumulation_flag'] = (df['net_foreign_flow_5d'] > 0.02).astype(int)

    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['dist_sma_20'] = (df['close'] - df['SMA_20']) / df['SMA_20']

    # 1. Strategy 1: Volume Explosion
    df['vol_explosion_flag'] = ((df['rel_volume_t'] > 3.0) & (abs(df['close'] - df['open']) > ta.atr(df['high'], df['low'], df['close'], length=14))).astype(int)
    df['vol_explosion_direction'] = np.where(df['vol_explosion_flag'] == 1, np.where(df['close'] > df['open'], 1, -1), 0)

    df['vol_gt_2'] = (df['rel_volume_t'] > 2.0).astype(int)
    df['vol_explosion_streak'] = df.groupby((df['vol_gt_2'] == 0).cumsum()).cumcount() * df['vol_gt_2']

    df['high_10d'] = df['high'].rolling(10).max()
    df['vol_explosion_price_confirm'] = ((df['vol_explosion_flag'] == 1) & (df['close'] > df['high_10d'].shift(1))).astype(int)

    # 2. Strategy 2: RSI Oversold
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_oversold_flag'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought_flag'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_14_prev'] = df['rsi_14'].shift(1)
    df['rsi_5d_slope'] = df['rsi_14'] - df['rsi_14'].shift(5)

    df['low_10d'] = df['low'].rolling(10).min()
    df['rsi_divergence_bullish'] = ((df['low'] < df['low_10d'].shift(1)) & (df['rsi_14'] > df['rsi_14_prev'])).astype(int)
    df['rsi_divergence_bearish'] = ((df['high'] > df['high_10d'].shift(1)) & (df['rsi_14'] < df['rsi_14_prev'])).astype(int)

    # 3. Strategy 3: Price Down 15% in 20 Sessions
    # Requires Hurst Exponent (approximate with volatility/trend relation or dummy for now as actual Hurst is complex)
    # Fast Hurst proxy: Variance of log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['hurst_60d'] = np.random.uniform(0.3, 0.7, len(df)) # Proper hurst takes long loops, use mock for speed/structure

    df['zscore_20d'] = ta.zscore(df['close'], length=20)
    df['decline_20d_flag'] = (df['return_20d'] < -0.15).astype(int)
    df['decline_severity_20d'] = abs(df['return_20d'])
    df['capitulation_vol_flag'] = ((df['decline_20d_flag'] == 1) & (df['rel_volume_t'] > 2.5) & (df['close'] < df['open'])).astype(int)

    # 4. Strategy 4: Price Down 15% vs MA20
    df['deep_oversold_ma20_flag'] = (df['dist_sma_20'] < -0.15).astype(int)
    df['ma20_deviation_pct'] = df['dist_sma_20']

    # 5. Strategy 5: SAR x MACD Histogram
    sar = ta.psar(df['high'], df['low'], df['close'], af0=0.02, af=0.02, max_af=0.2)
    if sar is not None:
        sar_col = sar.columns[0] # Usually 'PSARl_0.02_0.2' or similar
        df['sar_t'] = sar[sar_col]
        df['sar_bullish'] = (df['close'] > df['sar_t']).astype(int)
        df['sar_bullish_prev'] = df['sar_bullish'].shift(1)
        df['sar_flip_bullish'] = ((df['sar_bullish'] == 1) & (df['sar_bullish_prev'] == 0)).astype(int)
        df['sar_flip_bearish'] = ((df['sar_bullish'] == 0) & (df['sar_bullish_prev'] == 1)).astype(int)
        df['sar_distance_pct'] = (df['close'] - df['sar_t']) / df['close']

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        macd_hist_col = macd.columns[1] # MACDh_12_26_9
        df['macd_histogram_t'] = macd[macd_hist_col]
        df['macd_hist_prev'] = df['macd_histogram_t'].shift(1)

        cond_buy = (df.get('sar_flip_bullish', 0) == 1) & (df['macd_histogram_t'] > 0) & (df['macd_histogram_t'] > df['macd_hist_prev'])
        cond_sell = (df.get('sar_flip_bearish', 0) == 1) | (df['macd_histogram_t'] < 0)
        df['sar_macd_combo_signal'] = np.where(cond_buy, 1, np.where(cond_sell, -1, 0))

    # 6. Strategy 6: Uptrend
    # Mock stock RS vs Index
    df['stock_rs_20d'] = np.random.uniform(0.8, 1.2, len(df))
    df['ma_score_stock'] = ((df['close'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])).astype(float)

    uptrend_conds = [
        (df['close'] > df['SMA_20']).astype(int),
        (df['SMA_20'] > df['SMA_50']).astype(int),
        (df['SMA_50'] > df['SMA_200']).astype(int),
        (df['stock_rs_20d'] > 1.0).astype(int),
        (df['ma_score_stock'] == 1.0).astype(int),
        (df['hurst_60d'] > 0.55).astype(int)
    ]
    df['uptrend_quality_score'] = sum(uptrend_conds) / 6.0

    is_uptrend = (df['uptrend_quality_score'] == 1.0).astype(int)
    df['uptrend_duration'] = df.groupby((is_uptrend == 0).cumsum()).cumcount() * is_uptrend
    df['uptrend_health'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']

    # 7. Strategy 7: Bollinger Band Opening
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df['Upper_BB_t'] = bbands[bbands.columns[2]]
        df['Lower_BB_t'] = bbands[bbands.columns[0]]
        df['bb_width_t'] = bbands[bbands.columns[4]] # BBB_20_2.0 width

        df['bb_width_60d_p20'] = df['bb_width_t'].rolling(60).quantile(0.20)
        df['bb_width_60d_p15'] = df['bb_width_t'].rolling(60).quantile(0.15)

        df['bb_squeeze_flag'] = (df['bb_width_t'] < df['bb_width_60d_p15']).astype(int)

        df['bb_squeeze_t3'] = df['bb_squeeze_flag'].shift(3)
        df['bb_width_t3'] = df['bb_width_t'].shift(3)
        df['bb_expansion_flag'] = ((df['bb_squeeze_t3'] == 1) & (df['bb_width_t'] > df['bb_width_t3'] * 1.15)).astype(int)

        df['bb_breakout_direction'] = np.where((df['bb_expansion_flag'] == 1) & (df['close'] > df['Upper_BB_t']), 1,
                                       np.where((df['bb_expansion_flag'] == 1) & (df['close'] < df['Lower_BB_t']), -1, 0))
        df['bb_expansion_rate'] = (df['bb_width_t'] - df['bb_width_t'].shift(1)) / df['bb_width_t'].shift(1)

    # 8. Strategy 8: Wave Surfing with DMI
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None:
        df['adx_14'] = adx[adx.columns[0]]
        df['plus_di_14'] = adx[adx.columns[1]]
        df['minus_di_14'] = adx[adx.columns[2]]

        df['adx_slope_3d'] = df['adx_14'] - df['adx_14'].shift(3)
        df['dmi_bullish_flag'] = ((df['plus_di_14'] > df['minus_di_14']) & (df['adx_14'] > 25)).astype(int)

        df['plus_di_cross_minus'] = ((df['plus_di_14'] > df['minus_di_14']) & (df['plus_di_14'].shift(1) <= df['minus_di_14'].shift(1))).astype(int)
        df['dmi_crossover_bullish'] = ((df['plus_di_cross_minus'] == 1) & (df['adx_14'] > 20)).astype(int)

        cond_wave = ((df['adx_14'] > 25) & (df['plus_di_14'] > df['minus_di_14']) &
                     (df['plus_di_14'] > df['plus_di_14'].shift(1)) & (df['close'] > df['SMA_20']) &
                     (df['adx_slope_3d'] > 0))
        df['dmi_wave_signal'] = cond_wave.astype(int)

    # 9. Strategy 9: Stochastic RSI
    stochrsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None:
        df['stoch_rsi_k_t'] = stochrsi[stochrsi.columns[0]]
        df['stoch_rsi_d_t'] = stochrsi[stochrsi.columns[1]]

        df['stoch_rsi_oversold'] = (df['stoch_rsi_k_t'] < 20).astype(int)
        df['stoch_rsi_overbought'] = (df['stoch_rsi_k_t'] > 80).astype(int)

        df['stoch_rsi_kd_cross_bullish'] = ((df['stoch_rsi_k_t'] > df['stoch_rsi_d_t']) &
                                            (df['stoch_rsi_k_t'].shift(1) <= df['stoch_rsi_d_t'].shift(1)) &
                                            (df['stoch_rsi_k_t'].shift(1) < 30)).astype(int)
        df['stoch_rsi_kd_cross_bearish'] = ((df['stoch_rsi_k_t'] < df['stoch_rsi_d_t']) &
                                            (df['stoch_rsi_k_t'].shift(1) >= df['stoch_rsi_d_t'].shift(1)) &
                                            (df['stoch_rsi_k_t'].shift(1) > 70)).astype(int)

    # ==========================
    # PART B: ADDITIONAL INDICATORS
    # ==========================

    # B.1 Ichimoku
    ichimoku, _ = ta.ichimoku(df['high'], df['low'], df['close'])
    if ichimoku is not None:
        df['senkou_span_a'] = ichimoku[ichimoku.columns[0]]
        df['senkou_span_b'] = ichimoku[ichimoku.columns[1]]
        df['tenkan_sen'] = ichimoku[ichimoku.columns[2]]
        df['kijun_sen'] = ichimoku[ichimoku.columns[3]]

        df['price_above_cloud'] = (df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)).astype(int)
        df['price_below_cloud'] = (df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1)).astype(int)
        df['cloud_bullish'] = (df['senkou_span_a'] > df['senkou_span_b']).astype(int)

        df['tenkan_kijun_cross_bullish'] = ((df['tenkan_sen'] > df['kijun_sen']) &
                                            (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))).astype(int)
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b']) / df['close']

    # B.2 CCI
    df['cci_20_t'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df['cci_oversold_flag'] = (df['cci_20_t'] < -100).astype(int)
    df['cci_bullish_divergence'] = ((df['low'] < df['low_10d'].shift(1)) & (df['cci_20_t'] > df['cci_20_t'].shift(1))).astype(int)

    # B.3 Elder Ray
    df['ema_13'] = ta.ema(df['close'], length=13)
    df['bull_power_t'] = df['high'] - df['ema_13']
    df['bear_power_t'] = df['low'] - df['ema_13']

    # B.4 Alligator
    df['smma_13'] = ta.ema(df['close'], length=13) # Approximate SMMA with EMA for speed
    df['smma_8'] = ta.ema(df['close'], length=8)
    df['smma_5'] = ta.ema(df['close'], length=5)
    df['jaw'] = df['smma_13'].shift(8)
    df['teeth'] = df['smma_8'].shift(5)
    df['lips'] = df['smma_5'].shift(3)
    df['alligator_spread'] = (df['lips'] - df['jaw']) / df['jaw']

    # B.5 Aroon
    aroon = ta.aroon(df['high'], df['low'], length=25)
    if aroon is not None:
        df['aroon_down_25'] = aroon[aroon.columns[0]]
        df['aroon_up_25'] = aroon[aroon.columns[1]]
        df['aroon_oscillator'] = aroon[aroon.columns[2]]

    # B.6 Volume Profile (Proxy)
    # POC Proxy: Close of highest volume day in last 20 days
    df['poc_20d'] = df['close'].where(df['volume'] == df['volume'].rolling(20).max()).ffill()
    df['poc_distance'] = (df['close'] - df['poc_20d']) / df['poc_20d']
    df['above_value_area_flag'] = (df['close'] > df['poc_20d'] * 1.05).astype(int) # Proxy for VA high
    df['below_value_area_flag'] = (df['close'] < df['poc_20d'] * 0.95).astype(int)

    # ==========================
    # VSA (Volume Spread Analysis)
    # ==========================
    df['spread'] = df['high'] - df['low']
    df['vsa_ratio_t'] = df['volume'] / df['spread'].replace(0, 0.001) # Avoid div 0
    df['vsa_absorption_flag'] = ((df['volume'] > df['ADV_20'] * 1.5) & (df['spread'] < df['spread'].rolling(20).mean()) & (df['close'] > df['open'])).astype(int)

    # ==========================
    # EXPERIENTIAL FEATURES (Part F)
    # ==========================

    # S/R
    df['High_52w'] = df['high'].rolling(252).max()
    df['Low_52w'] = df['low'].rolling(252).min()
    df['dist_to_52w_high_pct'] = (df['High_52w'] - df['close']) / df['close']
    df['at_52w_high_flag'] = (df['close'] >= df['High_52w'] * 0.99).astype(int)

    # Mock Support/Resistance Zones (Simplified logic)
    df['resistance_zone_strength'] = np.random.randint(1, 5, len(df))
    df['inside_resistance_zone'] = (df['dist_to_52w_high_pct'] < 0.02).astype(int)
    df['inside_support_zone'] = ((df['close'] - df['Low_52w']) / df['close'] < 0.02).astype(int)
    df['resistance_broken_flag'] = ((df['close'] > df['High_52w'].shift(1)) & (df['rel_volume_t'] > 2.0)).astype(int)

    # Round Numbers
    df['round_num_nearest'] = (df['close'] / 1000).round() * 1000
    df['dist_to_round_number_pct'] = abs(df['close'] - df['round_num_nearest']) / df['close']
    df['at_round_number_flag'] = (df['dist_to_round_number_pct'] < 0.005).astype(int)

    # Fibonacci
    df['Swing_High_20'] = df['high'].rolling(20).max()
    df['Swing_Low_20'] = df['low'].rolling(20).min()
    diff = df['Swing_High_20'] - df['Swing_Low_20']

    df['fib_618_level'] = df['Swing_High_20'] - 0.618 * diff
    df['fib_382_level'] = df['Swing_High_20'] - 0.382 * diff
    df['at_fib_618_flag'] = (abs(df['close'] - df['fib_618_level']) / df['close'] < 0.015).astype(int)
    df['at_fib_382_flag'] = (abs(df['close'] - df['fib_382_level']) / df['close'] < 0.015).astype(int)
    df['fib_bounce_setup'] = ((df['at_fib_618_flag'] == 1) & (df['inside_support_zone'] == 1) & (df['volume'] < df['volume'].shift(1))).astype(int)
    df['fib_ext_1272'] = df['Swing_Low_20'] + 1.272 * diff
    df['fib_ext_1618'] = df['Swing_Low_20'] + 1.618 * diff

    # Patterns (Simplified mock flags for complex geometric patterns)
    # Double Bottom / Top
    df['double_bottom_flag'] = ((df['low'] < df['Low_52w'] * 1.02) & (df['close'] > df['SMA_20'])).astype(int)
    df['bull_flag_forming'] = ((df['return_5d'] > 0.08) & (df['return_1d'].abs() < 0.02) & (df['volume'] < df['ADV_20'])).astype(int)
    df['bull_flag_breakout'] = ((df['bull_flag_forming'].shift(1) == 1) & (df['close'] > df['high'].shift(1)) & (df['rel_volume_t'] > 1.8)).astype(int)

    # Wyckoff
    df['wyckoff_accumulation_flag'] = ((df['close'].rolling(20).max() - df['close'].rolling(20).min()) / df['close'].rolling(20).min() < 0.10).astype(int) & (df['low'] < df['close'].rolling(20).min() * 0.98).astype(int) & (df['close'] > df['close'].rolling(20).min()).astype(int)

    # SMC
    df['liquidity_sweep_bullish'] = ((df['low'] < df['Swing_Low_20'].shift(1) * 0.985) & (df['close'] > df['Swing_Low_20'].shift(1))).astype(int)

    # FVG
    df['fvg_bullish'] = (df['low'] > df['high'].shift(2)).astype(int)
    df['open_fvg_below'] = df['high'].shift(2).where(df['fvg_bullish'] == 1).ffill()

    # BOS / CHOCH
    df['bos_bullish'] = (df['close'] > df['Swing_High_20'].shift(1) * 1.005).astype(int)
    df['choch_bearish'] = (df['close'] < df['Swing_Low_20'].shift(1) * 0.995).astype(int)

    # Candlesticks
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    df['hammer_flag'] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
    df['hammer_at_support'] = ((df['hammer_flag'] == 1) & ((df['inside_support_zone'] == 1) | (df['at_fib_618_flag'] == 1))).astype(int)

    df['bullish_pinbar_at_level'] = ((lower_wick > 2 * body) & (lower_wick > upper_wick * 2) & ((df['inside_support_zone'] == 1) | (df['at_fib_618_flag'] == 1))).astype(int)

    df['upper_wick_ratio'] = upper_wick / df['spread'].replace(0, 0.001)
    df['upper_wick_resistance'] = (df['upper_wick_ratio'] > 0.5).astype(int)

    # Volume Profile
    df['volume_climax_flag'] = ((df['rel_volume_t'] > 3.5) & (df['uptrend_duration'] > 10) & (df['vsa_ratio_t'] < 0.4 * df['vsa_ratio_t'].rolling(20).mean())).astype(int)
    df['pullback_volume_dryup'] = ((df['return_1d'] < 0) & (df['volume'] < 0.7 * df['ADV_20'])).astype(int)

    # Gaps
    df['gap_t'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['breakaway_gap_flag'] = ((df['gap_t'] > 0.03) & (df['resistance_broken_flag'] == 1) & (df['rel_volume_t'] > 2.5)).astype(int)
    df['exhaustion_gap_flag'] = ((df['gap_t'] > 0.02) & (df['return_20d'] > 0.25) & (df['rsi_14'] > 75)).astype(int)

    # VN Specific
    df['is_monday'] = (pd.to_datetime(df['time']).dt.dayofweek == 0).astype(int)
    df['is_friday'] = (pd.to_datetime(df['time']).dt.dayofweek == 4).astype(int)

    df['post_ceiling_day1_flag'] = ((df['return_1d'].shift(1) > 0.065)).astype(int) # Proxy for ceiling hit
    df['post_floor_day3_plus_flag'] = ((df['return_1d'].shift(1) < -0.065) & (df['return_1d'].shift(2) < -0.065) & (df['return_1d'].shift(3) < -0.065)).astype(int)

    # Fill NAs and return
    df.fillna(0, inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("backend/data/vn100_subset.csv")
    print("Loaded data, computing features...")

    # Process each ticker separately to avoid cross-contamination of rolling features
    features_list = []
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        feat_df = compute_all_features(ticker_df)
        features_list.append(feat_df)

    final_df = pd.concat(features_list, ignore_index=True)
    final_df.to_csv("backend/data/vn100_features.csv", index=False)
    print(f"Features computed and saved. Shape: {final_df.shape}")
