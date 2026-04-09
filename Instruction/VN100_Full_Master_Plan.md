# VN100 Quant Model — Full Master Plan v2
## Model · Signals · Facebook Channel Operations

---

# PART A — NEW STRATEGIES FROM BACKTEST DATA

The following strategies are sourced from observed backtest win-rate data across Vietnamese market conditions. Each strategy has a documented **optimal holding period** (the T+X with the highest historical win rate). The model uses this data to determine which signal type to issue AND how long to hold.

---

## STRATEGY 1 — Volume Explosion (Bùng nổ khối lượng)
**Optimal Hold**: T+180 (61.4% win rate)
**Short-term note**: T+3 and T+10 are useful entry confirmation but don't hold — the real payoff is long-term.

### Definition
A volume explosion event occurs when today's volume is significantly higher than the recent baseline, signaling a major market participant has taken a position.

### Detection Rules
- `rel_volume_t > 3.0` (today's volume is 3× the 20-day average)
- AND the volume is accompanied by a directional close: `Close_t > Open_t` (green candle) for bullish, or `Close_t < Open_t` (red candle) for bearish
- AND the price move is at least 1 ATR: `abs(Close_t - Close_{t-1}) > ATR_14_t`

### Features to Add
- `vol_explosion_flag`: 1 if `rel_volume_t > 3.0`, else 0
- `vol_explosion_direction`: +1 if bullish explosion, -1 if bearish explosion, 0 if none
- `vol_explosion_streak`: Number of consecutive days with `rel_volume > 2.0` (sustained accumulation)
- `vol_explosion_price_confirm`: 1 if volume explosion is accompanied by a close above the 10-day high

### Signal Output for Facebook
> 🔥 **VOLUME EXPLOSION** detected on [STOCK]
> Entry zone: [price range]
> Target horizon: T+60 to T+180
> Stop-loss: Close below kalman_price − 1.5σ

---

## STRATEGY 2 — RSI Oversold (RSI quá bán)
**Optimal Hold**: T+60 (74.3% win rate) — this is the highest single win rate in the backtest table
**Short-term**: T+3 is weak (46.3%), so do NOT enter and flip quickly — you must be patient

### Definition
RSI has dropped below the oversold threshold, indicating the stock has been sold too aggressively and is statistically due for a bounce.

### Detection Rules
- **Entry**: `RSI_14 < 30`
- **Confirmation**: `RSI_14_t > RSI_14_{t-1}` (RSI is turning upward from oversold)
- **Filter**: `net_foreign_flow_5d > -0.05` (foreigners are not actively fleeing)
- **Filter 2**: `vnindex_ma_score > -0.67` (not in a full macro downtrend — at least one MA is positive)

### Enhanced RSI Rules
- `rsi_divergence_bullish`: Price makes a new 10-day low BUT RSI does NOT make a new low. Set flag = 1. This is a much stronger signal than RSI alone (hidden strength from institutions).
- `rsi_divergence_bearish`: Price makes a new 10-day high BUT RSI does NOT make a new high. Set flag = 1 (hidden weakness).
- `rsi_14_t`, `rsi_5d_slope`, `rsi_oversold_flag` (RSI < 30), `rsi_overbought_flag` (RSI > 70)

### Signal Output for Facebook
> 📉→📈 **RSI OVERSOLD BOUNCE** on [STOCK]
> RSI reading: [value] — turning up
> Best holding strategy: 60 days (74% historical win rate)
> Entry: [price range] | Stop: [lower bound]

---

## STRATEGY 3 — Price Down 15% in 20 Sessions (Giá giảm 15% trong 20 phiên)
**Optimal Hold**: T+5 (79.2% win rate) — a SHORT-TERM bounce play
**Logic**: After a severe short-term decline, price statistically reverts. This is a mean-reversion trade, not a trend trade.

### Detection Rules
- `return_20d < -0.15` (stock has dropped 15% or more in the last 20 sessions)
- AND `hurst_60d < 0.5` (mean-reverting regime — confirms bounce is likely)
- AND `zscore_20d < -1.5` (statistically oversold on z-score)
- AND `rel_volume_t > 1.5` on the last down day (capitulation selling = near the bottom)

### Features to Add
- `decline_20d_flag`: 1 if `return_20d < -0.15`
- `decline_severity_20d`: `abs(return_20d)` — how far the decline goes (the bigger, the stronger the bounce)
- `capitulation_vol_flag`: 1 if `decline_20d_flag = 1` AND `rel_volume > 2.5` AND today was a red candle (final flush)

### Signal Output for Facebook
> ⚡ **MEAN REVERSION BOUNCE** setup on [STOCK]
> Stock dropped [X]% in 20 sessions — statistical bounce zone
> Hold period: T+5 only (quick trade)
> Entry: [price] | Target: [upper bound] | Hard exit: T+5 close

---

## STRATEGY 4 — Price Down 15% vs MA20 (Giá giảm 15% so với MA20)
**Optimal Hold**: T+180 (100% win rate historically — the strongest long-term signal in the dataset)
**Note**: T+3 win rate is only 66.7%, so this is NOT a quick trade. This is a POSITION TRADE.

### Definition
The stock is trading 15% or more below its 20-day moving average. This is an extreme deviation that has historically always recovered over 180 days among VN100 quality stocks.

### Detection Rules
- `dist_sma_20 < -0.15` (price is 15%+ below SMA_20)
- AND the stock must be a VN100 constituent (quality filter — this strategy should NOT be applied to small caps)
- AND `net_foreign_flow_5d > 0` OR flat (institutions are not selling into the decline)
- AND `interbank_rate_change < 0.5%` (banking system is not in a liquidity crisis)

### Features to Add
- `deep_oversold_ma20_flag`: 1 if `dist_sma_20 < -0.15`
- `ma20_deviation_pct`: `(Close_t - SMA_20_t) / SMA_20_t` — continuous version
- `recovery_momentum`: After entry, track `(Close_t - entry_price) / entry_price` daily — for hold/exit decisions

### Signal Output for Facebook
> 🏆 **LONG-TERM VALUE BUY** on [STOCK]
> Trading [X]% below MA20 — extreme deviation
> Historical 180-day win rate: 100% on VN100
> Position trade: Hold 3–6 months | Partial exit at [upper bound]

---

## STRATEGY 5 — SAR × MACD Histogram (SAR x MACD Histogram)
**Optimal Hold**: T+20 (70.6% win rate)
**Logic**: A trend-following strategy that combines two independent indicators. Both must agree before a signal is issued.

### Parabolic SAR Rules
- `sar_t`: Computed using standard Wilder's SAR formula with `acceleration_factor = 0.02, max_acceleration = 0.2`
- `sar_bullish`: 1 if `Close_t > SAR_t` (price is above the SAR — uptrend mode), else 0
- `sar_flip_bullish`: 1 if `sar_bullish_t = 1` AND `sar_bullish_{t-1} = 0` (SAR just flipped from bearish to bullish — the entry signal)
- `sar_flip_bearish`: 1 if `sar_bullish_t = 0` AND `sar_bullish_{t-1} = 1` (SAR just flipped bearish — exit signal)

### Combined SAR + MACD Signal
- **Entry condition**: `sar_flip_bullish = 1` AND `macd_histogram_t > 0` AND `macd_histogram_t > macd_histogram_{t-1}` (histogram is positive and rising)
- **Exit condition**: `sar_flip_bearish = 1` OR `macd_histogram_t < 0`
- **Feature**: `sar_macd_combo_signal`: +1 for buy signal, -1 for sell/exit signal, 0 for neutral

### Features to Add
- `sar_t` (Parabolic SAR value)
- `sar_bullish` (binary direction)
- `sar_flip_bullish`, `sar_flip_bearish` (event flags)
- `sar_distance_pct = (Close_t - SAR_t) / Close_t` (how much buffer before SAR is hit)

### Signal Output for Facebook
> 📡 **SAR + MACD TREND SIGNAL** on [STOCK]
> Parabolic SAR flipped bullish + MACD histogram positive
> Hold: 20 trading days (~1 month)
> SAR stop-loss level: [SAR value]

---

## STRATEGY 6 — Uptrend (Uptrend)
**Optimal Hold**: T+180 (59.3% win rate)
**Note**: This is a pure trend-following strategy. Buy when the trend is clearly established and hold for the full run.

### Definition
A stock is in a confirmed uptrend when price, short-term MA, and long-term MA are all properly aligned.

### Detection Rules (All must be true simultaneously)
- `Close_t > SMA_20_t` (price above short MA)
- `SMA_20_t > SMA_50_t` (short MA above medium MA)
- `SMA_50_t > SMA_200_t` (medium MA above long MA — Golden arrangement)
- `stock_rs_20d > 1.0` (stock outperforming the index)
- `ma_score_stock = 1.0` (perfect MA alignment score)
- `hurst_60d > 0.55` (confirmed trending regime)

### Features to Add
- `uptrend_quality_score`: Sum of how many of the 6 conditions above are true, divided by 6. Range: 0 to 1.0. Only enter when score = 1.0.
- `uptrend_duration`: Number of consecutive days the uptrend has been in place (all 3 MAs aligned).
- `uptrend_health`: `(SMA_20 - SMA_50) / SMA_50` — the gap between MAs. A widening gap means the trend is accelerating; a narrowing gap is an early warning sign.

### Signal Output for Facebook
> 📈 **UPTREND CONFIRMED** on [STOCK]
> All MAs aligned perfectly — trend quality: 100%
> Trend duration: [X] days
> Hold: up to T+180 | Exit if MA20 crosses below MA50

---

## STRATEGY 7 — Bollinger Band Opening (Mở Band Bollinger)
**Optimal Hold**: T+5 (58.2% win rate, highest for this strategy)
**Logic**: When Bollinger Bands squeeze and then expand, a breakout is beginning. Capture the initial momentum move.

### Detection Rules
- **Squeeze Phase**: `bb_width_{t-5} < percentile(bb_width_60d, 20)` (the last 5 days had unusually narrow bands — coiling)
- **Expansion Trigger**: `bb_width_t > bb_width_{t-1} × 1.1` (bands are expanding today — breakout starting)
- **Direction**: If `Close_t > Upper_BB_t`: bullish breakout. If `Close_t < Lower_BB_t`: bearish breakdown.
- **Volume confirmation**: `rel_volume_t > 1.5` (breakout needs volume to be valid)

### Features to Add
- `bb_squeeze_flag`: 1 if `bb_width_t < percentile(bb_width_60d, 15)` — the coiling phase
- `bb_expansion_flag`: 1 if bands are expanding after a squeeze (`bb_squeeze_{t-3}` was 1 and now `bb_width_t > bb_width_{t-3} × 1.15`)
- `bb_breakout_direction`: +1 upper band breakout, -1 lower band breakout, 0 neutral
- `bb_expansion_rate`: `(bb_width_t - bb_width_{t-1}) / bb_width_{t-1}` — speed of expansion

### Signal Output for Facebook
> 🎯 **BOLLINGER BAND BREAKOUT** on [STOCK]
> Squeeze detected → bands now expanding
> Direction: [BULLISH / BEARISH]
> Short-term play: T+5 exit | Target: [price bound]

---

## STRATEGY 8 — Wave Surfing with DMI (Lướt sóng với DMI)
**Optimal Hold**: T+10 (70.0% win rate) — the most consistent short-to-medium term strategy
**Note**: Also strong at T+3 and T+5 (both 65%), meaning early entry is rewarded

### DMI / ADX Indicator
DMI (Directional Movement Index) and ADX (Average Directional Index) measure trend STRENGTH and DIRECTION separately. A strong trend moving in your direction is the ideal entry.

**Formulas:**
- `+DM = High_t - High_{t-1}` if positive and greater than `Low_{t-1} - Low_t`, else 0
- `-DM = Low_{t-1} - Low_t` if positive and greater than `High_t - High_{t-1}`, else 0
- `+DI_14 = 100 × EMA(+DM, 14) / ATR_14`
- `-DI_14 = 100 × EMA(-DM, 14) / ATR_14`
- `ADX_14 = 100 × EMA(|+DI - -DI| / (+DI + -DI), 14)`

**Interpretation:**
- `ADX > 25`: Strong trend exists (either direction)
- `ADX > 40`: Very strong trend
- `+DI > -DI`: Trend is upward (bullish)
- `-DI > +DI`: Trend is downward (bearish)

### Detection Rules for "Wave Surfing" Signal
- `ADX_14 > 25` (strong trend exists)
- AND `+DI_14 > -DI_14` (bullish direction)
- AND `+DI_14_t > +DI_14_{t-1}` (directional strength is increasing)
- AND `Close_t > SMA_20_t` (price above short-term MA)
- AND `ADX_slope = ADX_t - ADX_{t-3} > 0` (ADX is rising — trend is strengthening)

### Features to Add
- `plus_di_14`, `minus_di_14` (directional indicators)
- `adx_14` (trend strength, 0-100)
- `adx_slope_3d = adx_14_t - adx_14_{t-3}` (is trend accelerating or decelerating?)
- `dmi_bullish_flag`: 1 if `plus_di > minus_di` AND `adx > 25`
- `dmi_crossover_bullish`: 1 if `+DI` just crossed above `-DI` (entry signal, strongest with `adx > 20`)
- `dmi_wave_signal`: +1 if all wave surfing conditions met, else 0

### Signal Output for Facebook
> 🏄 **DMI WAVE SIGNAL** on [STOCK]
> ADX strength: [value] — trend is [STRONG / VERY STRONG]
> +DI: [value] > -DI: [value] — bullish direction confirmed
> Hold: T+10 | Exit if ADX drops below 20 or -DI crosses above +DI

---

## STRATEGY 9 — Price Rising + Stochastic RSI (Giá tăng và Stochastic RSI)
**Optimal Hold**: T+180 (53.2% win rate)
**Note**: This is the weakest strategy in the table at short holds. Only enter for long-term positioning.

### Stochastic RSI Formula
Stochastic RSI applies the Stochastic formula to the RSI value instead of price, making it more sensitive than either RSI or Stochastic alone.
- `stoch_rsi_k = (RSI_t - min(RSI, 14)) / (max(RSI, 14) - min(RSI, 14)) × 100`
- `stoch_rsi_d = EMA(stoch_rsi_k, 3)` (3-period smoothed version)

### Detection Rules
- `return_5d > 0.03` (price is trending up, minimum 3% in 5 days)
- AND `stoch_rsi_k < 50` (Stochastic RSI is NOT overbought — there is still room to run)
- AND `stoch_rsi_k > stoch_rsi_d` (K-line above D-line — momentum building)
- AND `adx_14 > 20` (some directional trend exists)

### Features to Add
- `stoch_rsi_k_t`, `stoch_rsi_d_t`
- `stoch_rsi_oversold`: 1 if `stoch_rsi_k < 20`
- `stoch_rsi_overbought`: 1 if `stoch_rsi_k > 80`
- `stoch_rsi_kd_cross_bullish`: 1 if K just crossed above D from below 30 (strong buy signal)
- `stoch_rsi_kd_cross_bearish`: 1 if K just crossed below D from above 70 (strong sell signal)

---

## UPDATED STRATEGY PRIORITY MATRIX

| Strategy | Best Hold | Win Rate | Signal Type | Regime |
|---|---|---|---|---|
| Price −15% vs MA20 | T+180 | 100% | Position Buy | Mean-revert (Hurst < 0.5) |
| RSI Oversold | T+60 | 74.3% | Swing Buy | Mean-revert |
| Price −15% in 20 sessions | T+5 | 79.2% | Bounce Trade | Mean-revert |
| DMI Wave Surfing | T+10 | 70.0% | Momentum Trade | Trending (Hurst > 0.55) |
| SAR × MACD | T+20 | 70.6% | Trend Trade | Trending |
| Bollinger Breakout | T+5 | 58.2% | Breakout Trade | Volatile |
| Volume Explosion | T+180 | 61.4% | Position Buy | Any |
| Uptrend | T+180 | 59.3% | Position Hold | Trending |
| Price Up + StochRSI | T+180 | 53.2% | Position Buy | Trending |

**Rule**: The model always selects the strategy with the highest applicable win rate for the current Hurst/regime. It issues only ONE primary signal per stock per day (the highest-conviction setup), plus a secondary signal if a second strategy is active simultaneously.

---

# PART B — ADDITIONAL INDICATORS

## B.1 Ichimoku Cloud (The Complete Picture)
Ichimoku is one of the most information-dense single indicators. It shows support/resistance, trend, and momentum simultaneously.

**Components:**
- `tenkan_sen = (highest high + lowest low) / 2` over 9 periods (Conversion Line)
- `kijun_sen = (highest high + lowest low) / 2` over 26 periods (Base Line)
- `senkou_span_a = (tenkan_sen + kijun_sen) / 2` plotted 26 periods ahead (Leading Span A)
- `senkou_span_b = (highest high + lowest low) / 2` over 52 periods, plotted 26 ahead (Leading Span B)
- `chikou_span = Close_t` plotted 26 periods back (Lagging Span)

**Features:**
- `price_above_cloud`: 1 if `Close_t > max(senkou_span_a, senkou_span_b)` — strong bullish
- `price_below_cloud`: 1 if `Close_t < min(senkou_span_a, senkou_span_b)` — strong bearish
- `cloud_bullish`: 1 if `senkou_span_a > senkou_span_b` (bullish cloud)
- `tenkan_kijun_cross_bullish`: 1 if tenkan just crossed above kijun (TK Cross — buy signal)
- `cloud_thickness = abs(senkou_span_a - senkou_span_b) / Close_t` — thick cloud = strong support/resistance

## B.2 Commodity Channel Index (CCI)
- **Formula**: `CCI_20 = (Typical_Price - SMA(Typical_Price, 20)) / (0.015 × Mean_Deviation)`
- `Typical_Price = (High + Low + Close) / 3`
- **Interpretation**: CCI > +100 = overbought, CCI < -100 = oversold. Unlike RSI, CCI has no upper/lower limit.
- **Features**: `cci_20_t`, `cci_oversold_flag` (CCI < -100), `cci_bullish_divergence` (CCI making higher lows while price makes lower lows)

## B.3 Elder Ray Index
- **Bull Power**: `High_t - EMA_13_t`
- **Bear Power**: `Low_t - EMA_13_t`
- **Features**: `bull_power_t`, `bear_power_t`
- **Signal**: Strong buy when `adx > 25` AND `bear_power_t < 0` AND `bear_power_t > bear_power_{t-1}` (bear power is negative but rising — bears are weakening)

## B.4 Williams Alligator
The Alligator uses three smoothed moving averages (SMMA) offset into the future to identify when a trend is emerging:
- `jaw = SMMA(Median_Price, 13)` shifted 8 bars forward (slowest, longest-term trend)
- `teeth = SMMA(Median_Price, 8)` shifted 5 bars forward
- `lips = SMMA(Median_Price, 5)` shifted 3 bars forward
- **Sleeping Alligator**: The three lines are intertwined — no clear trend (avoid trading)
- **Awakening Alligator**: Lines start to diverge — trend beginning
- **Eating Alligator**: Lines fully spread — strong trend in progress
- **Feature**: `alligator_spread = (lips - jaw) / jaw` — positive and widening = bullish trend accelerating

## B.5 Aroon Indicator
- **Aroon Up**: `100 × (25 - periods_since_25d_high) / 25`
- **Aroon Down**: `100 × (25 - periods_since_25d_low) / 25`
- **Aroon Oscillator**: `Aroon_Up - Aroon_Down`
- **Features**: `aroon_up_25`, `aroon_down_25`, `aroon_oscillator`
- **Signal**: `aroon_oscillator > 50` = strong uptrend beginning; `< -50` = strong downtrend

## B.6 Market Profile / Volume Profile (VN-Specific Implementation)
Instead of a full intraday volume profile (which requires tick data), approximate using daily data:
- **Point of Control (POC) proxy**: The close price of the highest-volume day in the last 20 days.
- `poc_distance = (Close_t - POC_20d) / POC_20d` — distance from the "fairest" price
- **Value Area**: The price range covering the top 70% of volume in the last 20 days (use High/Low of the 7 highest-volume days out of 10)
- **Features**: `poc_20d`, `above_value_area_flag`, `below_value_area_flag`
- **Why**: Stocks trading above the Value Area often revert to it; stocks below the Value Area find support there.

---

# PART C — ENHANCED BACKTEST FRAMEWORK

## C.1 Walk-Forward Validation (Anti-Overfitting)
Never use standard K-fold cross-validation on financial time-series. It causes data leakage. Use Walk-Forward instead.

**Process:**
1. Train on data from Day 1 to Day 750 (approximately 3 years)
2. Test on Day 751 to Day 875 (next ~6 months)
3. Slide window: Train on Day 1 to Day 875, test on Day 876 to Day 1000
4. Repeat until all data is consumed
5. Report the AVERAGE performance across all test windows (not the best single window)

**Minimum required**: 5 years of training data, 1 year of out-of-sample testing.

## C.2 Strategy Combination Rules (How to Stack Signals)
A "Tier 1 Signal" is when TWO or more independent strategies trigger simultaneously on the same stock on the same day.

**Tier Classification:**
- **Tier 1 (Highest confidence)**: 3+ strategies agree → Recommended position size: 100% of per-stock allocation
- **Tier 2 (High confidence)**: 2 strategies agree → 70% position size
- **Tier 3 (Single signal)**: 1 strategy only → 40% position size
- **Blocked**: Any signal where `bull_trap_flag = 1` OR `t25_risk > 1.0` → 0% (do not trade)

**Example Tier 1 setup:**
- `rsi_oversold_flag = 1` (RSI bounce setup)
- AND `decline_20d_flag = 1` (price down 15% in 20 sessions)
- AND `foreign_accumulation_flag = 1` (foreigners accumulating)
→ This is a 3-signal confluence: Tier 1 Buy

## C.3 Performance Metrics to Report (Beyond Win Rate)
The Facebook channel and internal model evaluation should track ALL of the following:
- **Win Rate**: % of closed trades that were profitable
- **Average Win**: Average % gain on winning trades
- **Average Loss**: Average % loss on losing trades
- **Profit Factor**: (Total Wins × Avg Win) / (Total Losses × Avg Loss). Must be > 1.5 to be viable.
- **Max Drawdown**: The worst peak-to-trough loss during the backtest period
- **Calmar Ratio**: Annualized return / Max Drawdown. Target > 1.0
- **Sharpe Ratio**: (Avg Return - Risk Free Rate) / Std(Returns). Target > 1.0
- **T+2.5 Adjusted Return**: Returns measured at T+3 (the first sellable day), not T+1

## C.4 The Backtesting Rules for VN Market Specifically
1. **Settlement delay**: All sell orders execute at T+3 minimum. No same-day or next-day flips.
2. **Price limit simulation**: Cap any single-day move at ±7%. If the theoretical exit price exceeds this, assume execution at the limit price.
3. **Slippage model**: Add 0.15% slippage to every entry and 0.15% to every exit (conservative estimate for VN100 liquidity).
4. **Commission**: Standard VN brokerage is 0.1% to 0.25% per trade. Use 0.2% total round-trip.
5. **Ceiling/Floor execution**: If the exit day has the stock at the floor (-7%), assume worst-case execution at floor price.
6. **Position sizing**: No single stock position exceeds 10% of portfolio (VN100 is concentrated in a few sectors).

---

# PART D — FACEBOOK CHANNEL OPERATION PLAN

## D.1 Channel Purpose & Positioning
**Channel Name suggestion**: "VN100 Tín Hiệu" or "VN Quant Signal" or "HOSE Radar"
**Positioning**: Professional daily stock signal page for VN100. Data-driven. No guessing. No hype.
**Target audience**: Retail Vietnamese investors who want institutional-quality signals in a simple format.
**Posting frequency**: 1 main post per day (evening, after market close 3:00 PM) + 1 morning brief post (8:00 AM before open)

## D.2 Daily Content Calendar

### MORNING POST (7:30–8:00 AM before market open)
**Title**: "⚡ Tín Hiệu Hôm Nay — [Date]"
**Content structure**:
1. Market environment snapshot (VN-Index MA score, breadth, foreign flow yesterday)
2. Top 3 stocks with active signals today (with entry zone, stop, target)
3. Watch list: stocks approaching a signal trigger
4. Risk warning if macro is negative

### AFTERNOON POST (3:15–4:00 PM after close)
**Title**: "📊 Tổng Kết Phiên — [Date]"
**Content structure**:
1. VN-Index summary: close, change, volume vs average
2. Signal performance: how did today's active signals perform?
3. New signals triggered today (detailed card for each)
4. Updated bounds for open positions

### WEEKLY POST (Sunday evening)
**Title**: "📅 Nhìn Lại Tuần + Tín Hiệu Tuần Tới"
**Content structure**:
1. Week performance recap for all active signals
2. Win rate of the week
3. Top setups forming for next week
4. Macro review (VN-Index trend, foreign flow weekly, USD/VND trend)

## D.3 Signal Card Format (for every individual stock signal)

Each signal posted to Facebook must include ALL of the following information:

```
═══════════════════════════════════
📌 [TICKER] — [COMPANY NAME]
Ngày: [Date] | Phiên: [Session]
═══════════════════════════════════

🔶 LOẠI TÍN HIỆU: [BUY / HOLD / SELL]
📐 CHIẾN LƯỢC: [Strategy Name]
⏰ THỜI GIAN NẮM GIỮ: T+[X] (đến ngày [date])

💰 VÙNG GIÁ
  Giá hiện tại: [Close_t]
  Vùng mua vào: [Lower_Entry - Upper_Entry]
  Mục tiêu (Upper Bound): [Upper_Bound]
  Cắt lỗ (Stop-Loss): [Stop_Loss_Price]
  Tỷ lệ Lợi/Rủi: [Expected_Return / Risk] = [ratio]

📊 CHỈ SỐ KỸ THUẬT
  RSI_14: [value] | ADX: [value]
  MACD Histogram: [value] (↑/↓)
  Khối lượng hôm nay: [X]× trung bình
  Dòng ngoại: [+ / - / flat] ([value]%)

🎯 ĐỘ TIN CẬY
  Xác suất tăng (mô hình): [X]%
  Số tín hiệu đồng thuận: [N]/9
  Rủi ro T+2.5: [LOW / MEDIUM / HIGH]
  Hạng tín hiệu: [TIER 1 / TIER 2 / TIER 3]

⚠️ ĐIỀU KIỆN HỦY TÍN HIỆU
  Thoát ngay nếu giá đóng cửa < [lower_bound]
  Thoát ngay nếu khối lượng sụt giảm + ngoại bán ròng

📈 Xác suất lịch sử ([Strategy]): [win_rate]% (T+[X])
═══════════════════════════════════
```

## D.4 Signal Types and Their Facebook Actions

| Signal | Post Color | Action for Followers |
|---|---|---|
| TIER 1 BUY | 🟢 Green card | Recommended entry in the buy zone |
| TIER 2 BUY | 🟡 Yellow card | Partial entry (50% position) |
| TIER 3 BUY | 🔵 Blue card | Watch only — wait for confirmation |
| HOLD | ⚪ Gray card | Do not add, do not sell |
| PARTIAL SELL | 🟠 Orange card | Sell 50% at upper bound, hold rest |
| FULL SELL / EXIT | 🔴 Red card | Exit entire position immediately |
| RISK WARNING | ⛔ Black card | Do not trade this stock — trap detected |

## D.5 Model Output → Facebook Post Pipeline

**Step 1**: Model runs every day at 3:30 PM after close.
**Step 2**: Model generates a JSON output for each VN100 stock:
```json
{
  "ticker": "HPG",
  "signal": "BUY",
  "tier": 1,
  "strategy": "RSI_OVERSOLD + VOLUME_EXPLOSION",
  "current_price": 26400,
  "entry_zone": [25800, 26600],
  "upper_bound": 29500,
  "stop_loss": 24900,
  "hold_days": 60,
  "direction_probability": 0.71,
  "signals_agreeing": 3,
  "t25_risk": "LOW",
  "rsi_14": 27.3,
  "adx_14": 31.2,
  "rel_volume": 2.8,
  "net_foreign_flow": 0.06,
  "garch_sigma_3d": 0.032,
  "historical_win_rate": 0.743,
  "cancel_condition_price": 24900
}
```
**Step 3**: A template engine converts the JSON into the formatted Facebook card.
**Step 4**: Human review (optional) before posting. Check for: earnings announcements, corporate news, extreme macro conditions.
**Step 5**: Post to Facebook page at scheduled time.

## D.6 Content for Non-Signal Days (When No Tier 1/2 Signals Exist)
On days when the model finds no high-quality entries:
- Post a "No signal day" post explaining WHY (e.g., "VN-Index in downtrend, macro score = -1.0, all signals filtered out")
- Post an educational piece about one of the 9 strategies
- Post a "Under the Hood" post showing how one feature (e.g., Hurst Exponent) works
- Post the weekly backtest performance update

**This is important for channel credibility**: A channel that says "no trade today" is MORE trustworthy than one that always has a signal.

## D.7 Transparency & Credibility Rules

To build trust as a public Facebook channel, follow these rules strictly:

1. **Post the signal BEFORE it happens**, never after. No retroactive "we predicted this."
2. **Track and publish ALL signals**, not just the winners. Monthly report with full P&L.
3. **Never claim guaranteed returns**. Always show historical win rates as probabilities, not guarantees.
4. **Show the stop-loss hit count**. A strategy with 70% win rate means 30% stop-outs. Show both.
5. **Post the model's regime status daily** so followers understand why signals are or aren't firing.
6. **Disclaimer on every post**: "Đây là tín hiệu từ mô hình định lượng, không phải lời khuyên đầu tư."

## D.8 Monthly Performance Report Format

Post at the start of each month summarizing the previous month:

```
═══ BÁO CÁO THÁNG [Month/Year] ═══

📊 TỔNG QUAN
Số tín hiệu phát ra: [N]
Tín hiệu Tier 1: [N] | Tier 2: [N] | Tier 3: [N]

✅ KẾT QUẢ
Win rate tổng: [X]%
Lợi nhuận trung bình (thắng): +[X]%
Lỗ trung bình (thua): -[X]%
Profit Factor: [X]

🏆 TOP SIGNALS THÁNG
1. [TICKER] +[X]% (chiến lược: [name])
2. [TICKER] +[X]%
3. [TICKER] +[X]%

❌ STOP-LOSS HIT
[TICKER] -[X]% | [TICKER] -[X]%

📐 THEO CHIẾN LƯỢC
DMI Wave: [X]% win rate | [N] signals
RSI Oversold: [X]% win rate | [N] signals
...

⚠️ GHI CHÚ
[Any unusual market conditions that affected performance]
═══════════════════════════════
```

---

# PART E — COMPLETE UPDATED FEATURE LIST (DELTA FROM V1)

Features added in this version (additions to the v1 spec):

| New Feature | Strategy Source | Type |
|---|---|---|
| `vol_explosion_flag` | Volume Explosion | Binary |
| `vol_explosion_direction` | Volume Explosion | Categorical |
| `vol_explosion_streak` | Volume Explosion | Integer |
| `rsi_divergence_bullish` | RSI Oversold | Binary |
| `rsi_divergence_bearish` | RSI Oversold | Binary |
| `decline_20d_flag` | Price -15%/20d | Binary |
| `decline_severity_20d` | Price -15%/20d | Continuous |
| `capitulation_vol_flag` | Price -15%/20d | Binary |
| `deep_oversold_ma20_flag` | Price -15% vs MA20 | Binary |
| `ma20_deviation_pct` | Price -15% vs MA20 | Continuous |
| `sar_t` | SAR×MACD | Continuous |
| `sar_bullish` | SAR×MACD | Binary |
| `sar_flip_bullish` | SAR×MACD | Binary (event) |
| `sar_flip_bearish` | SAR×MACD | Binary (event) |
| `sar_distance_pct` | SAR×MACD | Continuous |
| `sar_macd_combo_signal` | SAR×MACD | Categorical (-1/0/+1) |
| `uptrend_quality_score` | Uptrend | Continuous [0,1] |
| `uptrend_duration` | Uptrend | Integer |
| `uptrend_health` | Uptrend | Continuous |
| `bb_squeeze_flag` | Bollinger Breakout | Binary |
| `bb_expansion_flag` | Bollinger Breakout | Binary |
| `bb_breakout_direction` | Bollinger Breakout | Categorical |
| `bb_expansion_rate` | Bollinger Breakout | Continuous |
| `plus_di_14` | DMI Wave | Continuous |
| `minus_di_14` | DMI Wave | Continuous |
| `adx_14` | DMI Wave | Continuous |
| `adx_slope_3d` | DMI Wave | Continuous |
| `dmi_bullish_flag` | DMI Wave | Binary |
| `dmi_crossover_bullish` | DMI Wave | Binary (event) |
| `dmi_wave_signal` | DMI Wave | Categorical |
| `stoch_rsi_k_t` | StochRSI | Continuous |
| `stoch_rsi_d_t` | StochRSI | Continuous |
| `stoch_rsi_kd_cross_bullish` | StochRSI | Binary (event) |
| `stoch_rsi_kd_cross_bearish` | StochRSI | Binary (event) |
| `price_above_cloud` | Ichimoku | Binary |
| `cloud_bullish` | Ichimoku | Binary |
| `tenkan_kijun_cross_bullish` | Ichimoku | Binary |
| `cloud_thickness` | Ichimoku | Continuous |
| `cci_20_t` | CCI | Continuous |
| `cci_oversold_flag` | CCI | Binary |
| `cci_bullish_divergence` | CCI | Binary |
| `bull_power_t` | Elder Ray | Continuous |
| `bear_power_t` | Elder Ray | Continuous |
| `alligator_spread` | Williams Alligator | Continuous |
| `aroon_oscillator` | Aroon | Continuous |
| `poc_distance` | Volume Profile | Continuous |
| `above_value_area_flag` | Volume Profile | Binary |
| `signal_tier` | Meta | Categorical (1/2/3) |
| `signals_agreeing_count` | Meta | Integer |
| `optimal_hold_days` | Meta | Integer |
| `profit_factor_strategy` | Meta | Continuous |

**Total feature count after v2**: ~130 features per stock per day.


# Part F: Experiential Trading Knowledge
## "What a 20-Year Trader Knows Without Thinking"
### Encoded as Features, Rules, and Pattern Detectors

---

> **Preface for AI/Engineer**: This document translates *intuitive, experience-based* trading knowledge into explicit, computable rules. Every pattern here was originally discovered by traders watching price behavior for years. The goal is to encode that intuition into features the model can use. Where a veteran trader says *"I can feel the resistance"*, this document says *"compute this formula and set this flag"*.

---

## CHAPTER 1 — SUPPORT & RESISTANCE ZONES (Price Has Memory)

The single most important experiential concept: **price remembers where it has been**. Levels where price has previously reversed, stalled, or consolidated become magnets and walls in the future.

### 1.1 How to Identify Resistance Zones (Not Lines)
A resistance is NOT a single price. It is a ZONE — a range of prices where selling activity historically exceeded buying. Treat it as a band, not a line.

**Algorithm to detect resistance zones:**
1. Look back 252 trading days (1 year).
2. Find all "swing highs": a day where `High_t > High_{t-1}` AND `High_t > High_{t+1}` (local peak).
3. Cluster swing highs that are within 1.5% of each other into a single zone.
4. Score each zone by: `zone_strength = count_of_touches × recency_weight`
   - `recency_weight`: touches in the last 60 days count double, last 20 days count triple.
5. The top 3 scored zones above current price = active resistance zones.
6. The top 3 scored zones below current price = active support zones.

**Features:**
- `nearest_resistance_pct`: Distance from current price to nearest resistance zone top, as a percentage. `(resistance_zone_top - Close_t) / Close_t`
- `nearest_support_pct`: Distance to nearest support zone bottom.
- `resistance_zone_strength`: Score of nearest resistance (how many times price has tested it).
- `inside_resistance_zone`: 1 if current price is within a resistance zone band.
- `inside_support_zone`: 1 if current price is within a support zone band.

### 1.2 The Resistance Drop Rule (The Core Experiential Rule)
**What every trader knows**: *When price approaches a resistance zone it has tested before and failed, the probability of another rejection is high — especially on the first or second test after a previous rejection.*

**Rule encoding:**
- If `nearest_resistance_pct < 0.02` (price is within 2% of a resistance zone):
  - AND `resistance_zone_strength >= 3` (zone has been tested 3+ times)
  - AND `rel_volume_t < 1.2` (weak buying volume — no conviction)
  - → Set `resistance_rejection_risk = HIGH`
  - → **Reduce upper bound to resistance zone top**. Do not forecast a price above the zone as a near-term target.
  - → If already in a long position: reduce position size by 50%, move stop-loss up to entry price (lock in profit).

**The flip side — The Breakout Rule:**
- If price CLOSES above a resistance zone (not just touches it) with `rel_volume > 2.0`:
  - The resistance zone becomes a **new support zone** (role reversal).
  - → Set `resistance_broken_flag = 1`.
  - → The model should now use the OLD resistance as the new stop-loss level.

### 1.3 Historical High/Low Levels (The "Round Number" of Chart Levels)
- `dist_to_52w_high_pct = (High_52w - Close_t) / Close_t`
- `dist_to_52w_low_pct = (Close_t - Low_52w) / Close_t`
- `at_52w_high_flag`: 1 if `Close_t >= High_52w × 0.99` (within 1% of 52-week high)
- `at_alltime_high_flag`: 1 if price is within 2% of all-time high in the dataset
- **Rule**: When `at_52w_high_flag = 1` AND `rel_volume_t < 1.5`, expect resistance and possible rejection. When `at_52w_high_flag = 1` AND `rel_volume_t > 2.5` with a strong close, expect a **breakout continuation** (price in "blue sky" territory with no overhead resistance).

### 1.4 Previous Day High/Low (The Short-Term Memory)
Day traders and swing traders react to the previous day's high and low automatically.
- `prev_day_high = High_{t-1}`
- `prev_day_low = Low_{t-1}`
- `at_prev_high_flag`: 1 if `High_t >= prev_day_high × 0.998` (testing yesterday's high)
- `broke_prev_high_flag`: 1 if `Close_t > prev_day_high` AND `rel_volume > 1.3` (bullish)
- `broke_prev_low_flag`: 1 if `Close_t < prev_day_low` AND `rel_volume > 1.3` (bearish)
- **Rule**: A day that opens above yesterday's high and stays above = strong bullish continuation. A day that attempts yesterday's high but closes below = potential bull trap.

### 1.5 The "Round Number" Psychological Effect
In Vietnam (and all markets), traders anchor on round numbers. Prices like 10,000, 15,000, 20,000, 25,000, 30,000, 50,000, 100,000 VND act as psychological support and resistance.
- `dist_to_round_number_pct`: Distance to the nearest round number (multiples of 1,000 for stocks under 20,000; multiples of 5,000 for stocks 20,000–100,000; multiples of 10,000 for stocks above 100,000).
- `at_round_number_flag`: 1 if price is within 0.5% of a round number.
- **Rule**: First approach of a round number from below = likely resistance. After a clean break above a round number = the round number becomes support. Rebounds from round numbers below current price are entry opportunities.

---

## CHAPTER 2 — FIBONACCI: THE NATURAL PULLBACK LEVELS

Fibonacci retracement is not mysticism — it works because enough traders USE it, creating self-fulfilling levels. In the VN market, the 61.8% retracement is especially reliable.

### 2.1 Fibonacci Retracement Calculation
For every significant swing (from a major low to a major high, or vice versa):
- Swing Low (`A`) and Swing High (`B`) are detected as: the lowest close in 20 days and highest close in 20 days respectively.
- **Key retracement levels:**
  - 23.6%: `B - 0.236 × (B - A)` — shallow pullback, very strong trend
  - 38.2%: `B - 0.382 × (B - A)` — moderate pullback, healthy trend
  - 50.0%: `B - 0.500 × (B - A)` — midpoint (not Fibonacci but widely watched)
  - 61.8%: `B - 0.618 × (B - A)` — **The Golden Ratio. The most important level.**
  - 78.6%: `B - 0.786 × (B - A)` — deep retracement, trend may be weakening

**Features:**
- `fib_38_level`, `fib_50_level`, `fib_618_level` (computed daily from rolling 20-day swing)
- `dist_to_fib_618 = abs(Close_t - fib_618_level) / Close_t`
- `at_fib_618_flag`: 1 if `dist_to_fib_618 < 0.015` (within 1.5% of the 61.8% level)
- `at_fib_382_flag`: Same for 38.2% level
- `fib_confluence_flag`: 1 if a Fibonacci level and a support/resistance zone are within 1% of each other (extremely high probability level)

### 2.2 The Fibonacci Bounce Rule
**What experienced traders know**: *A stock that pulls back to the 61.8% Fibonacci level of its last major move, and then shows a bullish candle (hammer, engulfing, doji recovery) at that level with declining volume on the pullback, is one of the highest-probability long entries available.*

**Rule encoding:**
- Conditions: `at_fib_618_flag = 1` AND `inside_support_zone = 1` AND `volume_declining_pullback = 1`
- `volume_declining_pullback`: 1 if volume has been decreasing for the last 3 days of the pullback (selling pressure drying up)
- → Set `fib_bounce_setup = 1` (Tier 2 entry signal automatically)
- Combined with `rsi_oversold_flag = 1` → Tier 1 entry signal

### 2.3 Fibonacci Extension Targets
When a trade IS taken, use Fibonacci extensions to set profit targets:
- 127.2% extension: `A + 1.272 × (B - A)` — first target
- 161.8% extension: `A + 1.618 × (B - A)` — second target (golden extension)
- 261.8% extension: `A + 2.618 × (B - A)` — maximum target (rare, for explosive breakouts)
- **Feature**: `fib_ext_1272`, `fib_ext_1618` — these become the upper bound targets when `breakout_flag = 1`

---

## CHAPTER 3 — CHART PATTERNS (Geometric Price Memory)

Chart patterns are visual representations of the psychological battle between buyers and sellers. Each pattern has a predictable resolution probability.

### 3.1 Double Top / Double Bottom

**Double Top** (Bearish Reversal):
- Two peaks at approximately the same price level with a valley ("neckline") in between.
- Detection: Find two swing highs within 2% of each other, separated by at least 5 bars, with a swing low between them.
- The signal triggers when price **breaks below the neckline** (the swing low between the two peaks).
- `double_top_flag`: 1 if pattern detected AND `Close_t < neckline_price` (confirmation break)
- `double_top_target = neckline_price - (peak_price - neckline_price)` (measure rule: the drop equals the height of the pattern)
- **VN market note**: Double tops at the +7% ceiling zone are extremely powerful because of the price limit mechanics — two consecutive ceiling days with a gap between them is a textbook double top in VN context.

**Double Bottom** (Bullish Reversal):
- Mirror image. Two troughs at approximately the same level.
- `double_bottom_flag`: 1 if pattern detected AND `Close_t > neckline_price` (break above neckline)
- `double_bottom_target = neckline_price + (neckline_price - trough_price)`

### 3.2 Head and Shoulders (H&S)

The most reliable reversal pattern. Three peaks: left shoulder (lower), head (highest), right shoulder (lower, approximately equal to left). A neckline connects the two troughs between the shoulders.

**Detection:**
1. Find three swing highs where: `head_price > left_shoulder_price` AND `head_price > right_shoulder_price`
2. Left and right shoulders are within 3% of each other in height
3. The neckline is the straight line connecting the two troughs

**Signal**: `hs_pattern_flag = 1` when all three peaks are formed AND `Close_t < neckline_price` (the breakdown)
**Target**: `neckline_price - (head_price - neckline_price)` (measure rule)
**Volume rule**: Volume MUST decrease from left shoulder to head to right shoulder (declining conviction on each peak). If right shoulder volume is HIGHER than left shoulder, the pattern is suspect.

**Inverse H&S** (bullish): Mirror image — three troughs. Much more reliable than the regular H&S in bull markets.
- `inverse_hs_flag`: 1 on neckline breakout with increasing volume

### 3.3 Ascending/Descending Triangles

**Ascending Triangle** (Bullish continuation or reversal):
- Flat top (horizontal resistance, sellers defend the same price level repeatedly)
- Rising bottom (higher lows, buyers becoming more aggressive)
- Detection: `max(High, last 20d)` is flat (within 1.5%) while `min(Low, rolling 5d)` is making higher lows
- `ascending_triangle_flag`: 1 when pattern is detected
- `triangle_breakout_flag`: 1 when `Close_t > flat_top × 1.005` with `rel_volume > 1.5`
- **Experiential rule**: The more times price tests the flat top WITHOUT breaking, the MORE powerful the eventual breakout (compressed energy). 5+ touches of the same resistance = explosive breakout when it comes.

**Descending Triangle** (Bearish): Flat bottom, declining top.
- `descending_triangle_flag`, `triangle_breakdown_flag`

### 3.4 Bull Flag / Bear Flag (The Most Common Continuation Pattern)

**Bull Flag**: After a sharp upward move ("the pole"), price consolidates in a tight, slightly downward-drifting channel ("the flag") before resuming higher. This is the market "catching its breath."

**Detection:**
- "Pole": `return_5d > 0.08` (strong up move in 5 days, minimum 8%)
- "Flag": Next 3–10 days, price moves sideways or slightly down in a tight range: `(max_close_flag_period - min_close_flag_period) / min_close_flag_period < 0.05` (range less than 5%)
- Volume MUST decrease during the flag (less selling = the pullback is just profit-taking, not distribution)
- `bull_flag_forming`: 1 if pole detected and flag phase is active
- `bull_flag_breakout`: 1 if price closes above the upper trendline of the flag with `rel_volume > 1.8`
- **Target**: Pole height added to breakout point — `breakout_price + pole_height`

**Bear Flag**: Pole is a sharp DOWN move, flag drifts slightly upward, breakdown resumes.
- `bear_flag_forming`, `bear_flag_breakdown`

### 3.5 Wedge Patterns

**Rising Wedge** (Bearish signal despite rising price):
- Price is making higher highs AND higher lows, but the highs are rising SLOWER than the lows (converging lines).
- This means buyers are losing strength — each new high requires more effort.
- `rising_wedge_flag`: Detected when `upper_trendline_slope < lower_trendline_slope` over 15–20 days, both positive.
- **Rule**: A rising wedge that breaks DOWN with high volume is one of the most powerful short-term bearish signals. In VN market context, this often appears in the final week before a sector correction.

**Falling Wedge** (Bullish signal despite falling price):
- Mirror: both lines declining but converging, lows declining SLOWER than highs.
- `falling_wedge_flag`: Bullish reversal setup.

### 3.6 The "Three Pushes" Exhaustion Pattern (Wyckoff Principle)
**What traders know**: *Markets rarely reverse in one move. They usually make three attempts at a level — three pushes to a high, or three drops to a low — before reversing.*

Detection for "Three Pushes to a High" (bearish exhaustion):
- Three successive swing highs, each slightly higher than the last.
- BUT: Each push is accompanied by DECREASING volume AND DECREASING momentum (MACD histogram lower on each push).
- `three_pushes_high_flag`: 1 if three consecutive swing highs with declining volume and momentum.
- **Action**: After the third push, expect a sharp reversal. This is the FOMO peak. Smart money sold on the first push; the third push is retail FOMO.
- `three_pushes_low_flag`: Three declining lows with decreasing volume (exhaustion selling) — bullish reversal.

---

## CHAPTER 4 — WYCKOFF METHODOLOGY (Institutional Footprints)

Richard Wyckoff's framework describes how large institutions (the "Composite Man") accumulate and distribute stock. Understanding these phases gives retail traders a huge edge.

### 4.1 The Accumulation Phase (Institutions Buying Quietly)
Characteristics: Price trades in a range for an extended period. Volume is irregular. Price briefly drops below the range (a "Spring" or "Shakeout") to trigger retail stop-losses before reversing sharply.

**Detection algorithm:**
1. Price has been in a range for `N >= 20` days: `(max_close - min_close) / min_close < 0.10` (range less than 10%)
2. At least one "Spring": a day where `Low_t < range_low × 0.98` but `Close_t > range_low` (wick below, close back inside)
3. Post-spring: `net_foreign_flow_5d > 0.05` (institutions buying the spring)
4. Volume: Higher on up days than down days within the range (Wyckoff volume analysis)

**Feature**: `wyckoff_accumulation_flag`: 1 if all conditions met
**Action**: The "Spring" followed by a recovery close IS the entry. Stop-loss at the spring low.

**VN Market Implementation**: In Vietnam, the "Spring" is often engineered by market makers on the floor day (-7%). Retail panic-sells at the floor, institutions scoop up the shares, and the next day opens above the range. This is one of the most reliable setups in the VN100.

### 4.2 The Distribution Phase (Institutions Selling Quietly)
Mirror of accumulation. Price in a range after a long uptrend. Volume is high but price doesn't advance. An "Upthrust" (brief move above the range, followed by reversal) is the distribution equivalent of the Spring.

**Detection:**
1. Price range after an uptrend: preceded by `return_60d > 0.20` (came from a strong uptrend)
2. Range phase: `N >= 15` days of sideways action
3. "Upthrust": `High_t > range_high × 1.02` but `Close_t < range_high` (wick above, close back inside — a failed breakout)
4. `net_foreign_flow_5d < -0.03` (foreigners selling during the upthrust)

**Feature**: `wyckoff_distribution_flag`: 1 if all conditions met
**Action**: DO NOT buy. If holding, the upthrust IS the exit. Stop-loss above the upthrust high.

### 4.3 Phase Detection Using Simplified Wyckoff Scoring
Assign the stock to one of five phases daily:
- **Phase A (Stopping the Prior Trend)**: High volume reversal after a trend. `vol_explosion_flag = 1` AND trend reversal candle.
- **Phase B (Building the Cause)**: Sideways trading, testing both sides of range.
- **Phase C (Testing — Spring or Upthrust)**: The shakeout. `at_support_zone = 1` AND `spring_detected = 1`.
- **Phase D (Trend Begins)**: Price starts making consistent progress in one direction with supporting volume.
- **Phase E (Trend in Progress)**: Sustained trend, clean structure.

**Feature**: `wyckoff_phase` (categorical: A/B/C/D/E) — feed as encoded feature into the model.

---

## CHAPTER 5 — SUPPLY & DEMAND ZONES (Modern Version of S/R)

Supply and demand zones are more granular than traditional support/resistance. They identify the specific CANDLES where institutional buying or selling occurred.

### 5.1 Demand Zones (Where Institutions Bought)
A demand zone is formed when price rapidly LEFT a level (a sharp up move), meaning buyers at that level were so aggressive they overwhelmed all sellers quickly. The "unfilled orders" from institutions remain at that level.

**Detection:**
1. Find a "Base" candle: a small-bodied candle (body < 30% of the candle range).
2. Followed by a "Explosion" candle: the next 1–2 candles move at least 2× ATR upward with high volume.
3. The demand zone = the body range of the base candle (or the last 1–3 candles before the explosion).
4. The zone STAYS active until price returns to it and reacts (tests the demand zone).

**Feature**: `nearest_demand_zone_low`, `nearest_demand_zone_high` (price boundaries)
**Feature**: `inside_demand_zone`: 1 if current price is within an identified demand zone
**Feature**: `demand_zone_age`: How many days since the demand zone was formed (fresher = stronger)
**Rule**: The FIRST return to a demand zone is the highest probability entry. Second test is still valid. Third test means the zone is losing strength.

### 5.2 Supply Zones (Where Institutions Sold)
Mirror of demand: a sharp DOWN move from a base candle level. The supply zone is where institutions had large sell orders filled.
**Feature**: `inside_supply_zone`, `nearest_supply_zone_low`, `nearest_supply_zone_high`
**Rule**: When price re-enters a supply zone, SELL or avoid longs. The probability of rejection is highest on the first re-test.

### 5.3 The "Tested Once = Weaker" Rule
Each time a supply/demand zone is tested and price bounces off it, the zone gets WEAKER (unfilled orders are being consumed). After 3 tests, the zone is largely consumed and a breakout is likely.
- `zone_test_count`: How many times price has entered and exited the zone
- **Rule**: On first test of zone → high probability reaction. On third test → HIGH probability of breakout through the zone. Trade the breakout, not the reaction.

---

## CHAPTER 6 — SMART MONEY CONCEPTS (SMC) — Advanced

SMC is the modern framework for understanding HOW large institutions manipulate price before making their real moves. These are the patterns that veteran VN traders talk about in private.

### 6.1 Liquidity Sweeps ("Stop Hunting")
**What every experienced trader knows**: *Before a major move up, the market first drops to sweep the stop-losses of all the retail longs (placed just below the last swing low). After the stop-hunt is complete, the real move begins.*

**Detection:**
1. A recent swing low exists at price level `S` (the level where retail stop-losses cluster).
2. Price briefly drops BELOW `S` (intraday or on a daily close): `Low_t < S × 0.985`
3. Price then RECOVERS and closes ABOVE `S` on the same day or next day.
4. Volume is HIGH during the sweep (institutions are buying the stops being triggered).

**Feature**: `liquidity_sweep_bullish`: 1 if above conditions met within last 3 days (price swept below a swing low and recovered).
**Feature**: `liquidity_sweep_bearish`: Price swept above a swing high then reversed.
**Action**: A bullish liquidity sweep is a HIGH-CONVICTION buy entry. The smart money just collected all the stop-losses and is ready to move price up. Stop-loss: just below the sweep low.

### 6.2 Order Blocks
An order block is the LAST up-candle (bullish OB) or down-candle (bearish OB) before a large, impulsive move in the opposite direction. It marks where institutional orders were placed.

**Bullish Order Block detection:**
1. A series of down candles, then a large up-move of at least 3× ATR.
2. The bullish order block = the body of the LAST down candle before the big up move.
3. When price returns to this level, it acts as strong support.

**Feature**: `bullish_ob_zone_low`, `bullish_ob_zone_high`
**Feature**: `inside_bullish_ob`: 1 if price is currently within a bullish order block zone
**Bearish OB**: Last up-candle before a major down move. Acts as resistance when revisited.

**VN Implementation**: Order blocks formed on ceiling days are particularly strong. The day before a stock hits the +7% ceiling, the order block from that day becomes a very reliable support zone.

### 6.3 Fair Value Gaps (FVG) — The Magnet Effect
**What traders know**: *When price moves so fast that a "gap" is created between candles (the high of candle 1 is below the low of candle 3, with candle 2 being the explosive candle), that gap is almost always "filled" at some point — price returns to cover the gap.*

**Fair Value Gap detection:**
- `fvg_bullish`: `Low_{t} > High_{t-2}` (candle 3's low is higher than candle 1's high — a gap upward)
  - The FVG zone = `[High_{t-2}, Low_t]`
- `fvg_bearish`: `High_t < Low_{t-2}` (gap downward)
  - The FVG zone = `[High_t, Low_{t-2}]`

**Feature**: `open_fvg_above` (nearest upward FVG above current price — acts as resistance AND target for fill)
**Feature**: `open_fvg_below` (nearest downward FVG below — acts as support AND likely to be filled)
**Feature**: `dist_to_nearest_fvg_pct`

**Rule**: After a strong up-move that leaves a bullish FVG, when price pulls back into the FVG zone, that is a high-probability long entry (the FVG acts as support). The FVG is the model's "magnet target" — price always gravitates toward open FVGs.

### 6.4 Break of Structure (BOS) and Change of Character (CHoCH)
These concepts track the TREND at a structural level — not using averages, but using swing points.

**Market Structure (Bullish):**
- A series of Higher Highs (HH) and Higher Lows (HL) = bullish structure.
- `market_structure_bullish`: 1 if last 3 swing highs are each higher than the previous, AND last 3 swing lows are each higher than the previous.

**Break of Structure (BOS):**
- In a bullish trend: BOS = price breaks ABOVE the most recent swing high. Bullish continuation.
- `bos_bullish`: 1 if `Close_t > most_recent_swing_high × 1.005` (clean break above structure)

**Change of Character (CHoCH) — The Reversal Warning:**
- In a bullish trend: CHoCH = price breaks BELOW the most recent swing LOW (this is the first sign the trend may be ending).
- `choch_bearish`: 1 if `Close_t < most_recent_swing_low × 0.995` in a bullish trend.
- **Rule**: CHoCH does NOT immediately mean sell everything, but it means STOP buying and tighten stop-losses. A second CHoCH confirms the reversal.

---

## CHAPTER 7 — CANDLESTICK CONTEXT RULES (Pattern + Location = Signal)

**Critical rule**: A candlestick pattern means NOTHING on its own. It only matters AT a key level (support, resistance, Fibonacci, order block). A hammer candle in the middle of a range is noise. A hammer candle at the 61.8% Fibonacci support, after a volume decline pullback, inside a demand zone = one of the most powerful signals that exists.

### 7.1 Hammer / Shooting Star at Key Levels
**Hammer** (bullish reversal): Long lower wick (at least 2× the body length), small body near the top of the candle, appears at SUPPORT.
- `hammer_at_support`: 1 if `hammer_flag = 1` AND (`inside_support_zone = 1` OR `at_fib_618_flag = 1` OR `inside_demand_zone = 1`)

**Shooting Star** (bearish reversal): Long upper wick, small body near the bottom, appears at RESISTANCE.
- `shooting_star_at_resistance`: 1 if `shooting_star_flag = 1` AND (`inside_resistance_zone = 1` OR `inside_supply_zone = 1` OR `at_52w_high_flag = 1`)

### 7.2 Pinbar Rules (The Most Powerful Single-Candle Signal)
A pinbar (or pin bar) is a candle with a long wick on one side, indicating a sharp price rejection. The wick shows WHERE the market was rejected.

**Bullish pinbar:**
- `lower_wick > 2 × body` AND `lower_wick > upper_wick × 2`
- The shadow must "stick out" below the surrounding candles (piercing a level)

**Bearish pinbar:**
- `upper_wick > 2 × body` AND `upper_wick > lower_wick × 2`
- Pierces above a resistance level

**Feature**: `bullish_pinbar_at_level`: 1 if bullish pinbar occurs within a support/demand/Fibonacci zone
**Feature**: `bearish_pinbar_at_level`: 1 if bearish pinbar occurs within a resistance/supply/Fibonacci zone

### 7.3 Evening Star / Morning Star (3-Candle Reversal at Key Levels)
**Evening Star** (bearish, 3-candle):
- Candle 1: Large bullish candle (strong up day)
- Candle 2: Small-bodied candle (doji or near-doji) — indecision at the top
- Candle 3: Large bearish candle that closes well into candle 1's body

**Rule**: Only trade the Evening Star when it forms AT a resistance zone. Otherwise it is noise.
- `evening_star_at_resistance`: 1 if pattern detected AND `inside_resistance_zone = 1`

**Morning Star** (bullish, 3-candle): Mirror image.
- `morning_star_at_support`: 1 if pattern detected AND `inside_support_zone = 1`

### 7.4 The "Wick Fill" Pattern (The Next-Day Expectation)
**What traders know**: *When a candle has a very long upper wick (price went high but closed much lower), the next day often opens near the close and tries to fill that wick — moving UP toward the high of the previous candle. The reverse applies to long lower wicks.*

**Feature**: `upper_wick_fill_potential`:
- `upper_wick_size = High_t - max(Open_t, Close_t)`
- `upper_wick_ratio = upper_wick_size / (High_t - Low_t)`
- If `upper_wick_ratio > 0.5` (more than half the day's range was an upper wick that got rejected), set `upper_wick_resistance = 1` (next day likely to struggle at the wick high)
- If `upper_wick_ratio < 0.3` AND `lower_wick_ratio < 0.3` (clean close near the high), set `momentum_continuation = 1` (next day likely to continue)

---

## CHAPTER 8 — VOLUME PROFILE EXPERIENTIAL RULES

### 8.1 The "Volume Climax" Reversal
**What traders know**: *The highest volume day in a trend is often the LAST day of the trend. When everyone who wants to buy has bought (volume climax), there are no more buyers and price reverses.*

**Detection:**
- `volume_climax_flag`: 1 if `rel_volume_t > 3.5` AND the stock has been trending for at least 10 days (`uptrend_duration > 10`)
- AND `vsa_ratio_t < 0.4 × mean(vsa_ratio, 20)` (huge volume, narrow spread — distribution)
- **Rule**: Volume climax on an UP trend = bearish. Volume climax on a DOWN trend (panic capitulation) = bullish reversal. Context is everything.

### 8.2 "No Volume" Advance — The Warning
**What traders know**: *If a stock makes new highs on lower-than-average volume, the move is not backed by real buying. It is a "phantom" rally and is extremely vulnerable to reversal.*

- `low_vol_advance_flag`: 1 if `rel_volume_t < 0.7` AND `return_1d > 0.02` (price up 2%+ on below-average volume)
- **Rule**: A `low_vol_advance_flag` at or near a resistance zone is an extremely strong signal to NOT enter. Combined with `at_52w_high_flag`, this becomes a near-certain resistance rejection setup.

### 8.3 Volume Dry-Up During Pullback (The Bullish Reset)
**What traders know**: *A healthy pullback in an uptrend is characterized by DECLINING volume. This means the selling is light — just profit-taking, not distribution. When volume dries up on the pullback, it's the market saying "we're just pausing before the next leg up."*

- `pullback_volume_dryup`: Computed over the last N days where price is declining (`return_1d < 0`):
  - 1 if average volume during declining days in the last 5 days is `< 0.7 × ADV_20`
- **Rule**: `pullback_volume_dryup = 1` in an uptrend + at a support level = strong "add to position" signal.

---

## CHAPTER 9 — GAP THEORY (Opening Price Behavior)

### 9.1 Gap Types and Expected Behavior
**Breakaway Gap**: Occurs when price gaps above a well-established resistance zone on high volume. Usually NOT filled for weeks or months. Indicates a major shift.
- `breakaway_gap_flag`: 1 if `gap_t > 0.03` (gap up more than 3%) AND `broke_resistance_flag = 1` AND `rel_volume > 2.5`
- **Rule**: Do NOT fade a breakaway gap. The gap becomes support. Buy the first pullback TO the top of the gap.

**Runaway Gap** (Continuation Gap): Occurs in the middle of a strong trend. Signals trend acceleration.
- `runaway_gap_flag`: 1 if `gap_t > 0.02` in the direction of the existing trend AND `uptrend_quality_score > 0.8`
- **Rule**: Signals approximately the midpoint of the trend move. Strong continuation signal.

**Exhaustion Gap**: Occurs at the END of a trend — a final burst of enthusiasm before reversal. Looks like a breakaway gap but occurs after an already extended move.
- `exhaustion_gap_flag`: 1 if `gap_t > 0.02` AND `return_20d > 0.25` (stock already up 25%+ in 20 days) AND `rsi_14 > 75`
- **Rule**: Fade this gap. The stock is in FOMO territory. This gap WILL be filled within 5–10 days.

**Common Gap**: Small gap (< 2%) with no volume surge, not at any key level. Nearly always filled within 2–3 days.
- `common_gap_flag`: 1 if `0.005 < abs(gap_t) < 0.015` AND `rel_volume < 1.3`
- **Rule**: Gaps of this type return to the previous close within 2–3 sessions statistically. Use this as a short-term bound: the common gap level is a magnetic target.

### 9.2 The "Gap and Trap" (VN-Specific)
In Vietnam, due to the ±7% daily limit, a gap-up opening that immediately runs toward the ceiling often traps late buyers. The ceiling limits the upside but not the downside risk.
- `gap_trap_flag`: 1 if `gap_t > 0.04` (large gap up) AND `dist_to_ceiling_pct < 0.02` (near the ceiling) AND this is NOT a ceiling demand continuation (no `ceiling_gap_up_signal` from prior day)
- **Rule**: Avoid entries on gap-trap days. The stock will likely open, run briefly toward ceiling, then reverse — the late buyers who chased the gap are trapped.

---

## CHAPTER 10 — MARKET TIMING & SEASONAL PATTERNS (VN-Specific)

### 10.1 The "Monday Effect" and "Friday Effect"
- Vietnamese retail investors tend to be more optimistic on Mondays (fresh week, new hopes) and profit-take on Fridays.
- `is_monday`: 1 if trading day is Monday
- `is_friday`: 1 if trading day is Friday
- `day_of_week_effect`: Feature encoding day of week (0=Mon to 4=Fri). The model will learn that certain strategies (momentum) work better mid-week; reversal setups at Friday's close.

### 10.2 The Post-Ceiling Momentum Effect (VN-Unique)
**What VN traders know**: *When a stock hits the +7% ceiling today AND has high ceiling demand (unfilled orders), it almost always gaps up or runs to another ceiling or near-ceiling the next day. The momentum lasts 1–3 days. But after 3 consecutive ceiling days, the probability of a reversal increases dramatically.*

- `post_ceiling_day1_flag`: 1 if stock hit ceiling yesterday AND ceiling demand > 0.5 (strong momentum expected)
- `post_ceiling_day2_flag`: 1 if stock hit ceiling for 2 consecutive days
- `post_ceiling_day3_plus_flag`: 1 if 3+ consecutive ceiling days (REVERSAL RISK HIGH — smart money distributing)
- **Rule**: Day 1 after ceiling: continuation bias. Day 2: still positive but watch volume. Day 3+: switch to sell bias. This is one of the most reliable patterns in VN market behavior.

### 10.3 The Post-Floor Capitulation Effect
Mirror of the ceiling effect:
- `post_floor_day1_flag`: Stock hit -7% floor yesterday (bounce likely)
- `post_floor_day2_flag`: Two consecutive floor days (crisis or opportunity — check foreign flow)
- `post_floor_day3_plus_flag`: Three+ consecutive floors (EXTREME FEAR — highest bounce probability but also highest risk)
- **Rule**: If `post_floor_day3_plus_flag = 1` AND `net_foreign_flow_t > 0.05` (foreigners buying the panic): HIGHEST probability bounce setup in the entire VN playbook.

### 10.4 T+2.5 Behavioral Patterns (Settlement Psychology)
Because buyers are locked in for 2.5 days, specific behavioral patterns emerge:
- **The T+3 Dump**: Stocks that had a big move on Day T often see profit-taking pressure on Day T+3 (first day sellers can exit). Feature: `t3_exit_pressure`: 1 if the stock had `return > 0.04` 3 trading days ago AND `rel_volume_t > 1.5` today.
- **The "Lock-Up" Premium**: During a strong uptrend, buyers accept paying a small premium because they know the stock will be higher when they can sell. This inflates momentum. The premium disappears when the trend weakens.

### 10.5 Earnings Season Behavioral Pattern (VN)
Vietnamese companies report quarterly earnings during specific periods (mid-January, mid-April, mid-July, mid-October). During these windows:
- Stocks with positive earnings surprises gap up and tend to CONTINUE for 3–5 sessions.
- Stocks with negative surprises continue DOWN for 3–5 sessions (no quick bounce — sellers are determined).
- The model should widen bounds by 30% during earnings season (±5 trading days from report date).
- `earnings_season_flag`: 1 during the two-week earnings window each quarter.

---

## CHAPTER 11 — SECTOR ROTATION EXPERIENTIAL RULES (VN-Specific)

### 11.1 The Classic Vietnam Rotation Cycle
**What VN fund managers know**: The HOSE has a clockwork rotation pattern that repeats roughly every 6–18 months:
1. **Banks lead** (large caps, first to recover, driven by credit growth narrative)
2. **Securities firms follow** (brokerage stocks rise when retail volume increases — a 2nd-order effect of a bull market)
3. **Real estate joins** (property developers rise on credit expansion and bank lending)
4. **Steel and industrials catch up** (late cyclicals, driven by infrastructure spending)
5. **Consumer/retail peaks** (last to move, often marks the end of the bull cycle)

**Feature**: `rotation_stage` (1–5, estimated from relative strength ranking of each sector vs. VN-Index over the last 20 days)
**Rule**: Buy stocks in the NEXT sector in the rotation sequence when the current sector is overbought. Example: if Banks are at RSI > 70 and Securities RS is just turning positive, rotate into Securities.

### 11.2 Sector Leader vs. Laggard
Within each sector, identify the leader (strongest RS) and laggard (weakest RS).
- **Rule 1**: The sector leader breaks out FIRST. Other sector stocks follow 2–5 days later. When VCB (the banking leader) breaks above resistance, buy other bank stocks immediately — they will follow within a week.
- **Rule 2**: If the sector laggard is the ONLY one not participating in a sector rally after 5+ days, it may have a specific negative catalyst. Do NOT buy the laggard expecting catch-up.
- `is_sector_leader`: 1 if this stock has the highest `stock_rs_20d` in its sector
- `is_sector_laggard`: 1 if this stock has the lowest `stock_rs_20d` in its sector
- `sector_leader_broke_resistance`: 1 if the sector leader broke a resistance zone today (signal to buy laggards)

---

## CHAPTER 12 — THE COMPLETE DECISION TREE (How All Knowledge Combines)

This is how the model uses ALL the above knowledge when evaluating whether to issue a BUY signal:

```
STEP 1 — MACRO FILTER (if any fail → no trade)
  ✓ vnindex_ma_score > -0.33 (not in strong downtrend)
  ✓ breadth_pct_above_50sma > 0.35 (at least 35% of stocks healthy)
  ✓ interbank_rate_change < 1.0% (no liquidity crisis)
  ✓ usdvnd_return_10d < 0.03 (VND not rapidly depreciating)

STEP 2 — REGIME CHECK
  → IF hurst_60d > 0.55: Use MOMENTUM strategies (DMI, Uptrend, BOS)
  → IF hurst_60d < 0.45: Use MEAN REVERSION strategies (RSI, Fibonacci, double bottom)
  → IF hurst_60d 0.45–0.55: Reduce all position sizes by 50%

STEP 3 — LEVEL CONTEXT (Required for high-conviction trades)
  ✓ NOT inside_supply_zone AND NOT inside_resistance_zone
  ✓ OR: resistance_broken_flag = 1 (confirmed breakout)
  ✓ inside_support_zone OR inside_demand_zone OR at_fib_618_flag

STEP 4 — PATTERN CONFIRMATION (at least ONE required)
  ✓ hammer_at_support OR morning_star_at_support
  ✓ OR bullish_pinbar_at_level
  ✓ OR double_bottom_flag OR inverse_hs_flag
  ✓ OR fib_bounce_setup
  ✓ OR wyckoff_accumulation_flag (spring detected)
  ✓ OR liquidity_sweep_bullish
  ✓ OR bull_flag_breakout

STEP 5 — VOLUME / SMART MONEY CONFIRMATION (at least ONE required)
  ✓ vsa_absorption_flag = 1 (institutional absorption at support)
  ✓ OR foreign_accumulation_flag = 1
  ✓ OR pullback_volume_dryup = 1 (healthy pullback)
  ✓ OR post_floor_day1_flag = 1 with foreign buying

STEP 6 — STRATEGY SIGNAL CHECK (at least ONE required from the 9 backtested strategies)

STEP 7 — RISK GATES (if any fail → no trade)
  ✓ bull_trap_flag = 0
  ✓ distribution_at_resistance = 0
  ✓ exhaustion_gap_flag = 0
  ✓ three_pushes_high_flag = 0
  ✓ wyckoff_distribution_flag = 0
  ✓ upper_wick_resistance = 0 (if at a key level)
  ✓ t25_risk < 1.0
  ✓ choch_bearish = 0 (no change of character)

RESULT:
  Steps 1–7 all pass → TIER 1 signal
  Steps 1,2,3,6,7 pass but not all of 4,5 → TIER 2 signal
  Only Steps 1,2,6,7 pass → TIER 3 signal (watch only)
  Any gate in Step 1 or 7 fails → NO TRADE
```

---

## CHAPTER 13 — NEW FEATURES ADDED (Delta from Parts A–E)

| Feature | Concept | Type |
|---|---|---|
| `resistance_zone_strength` | S/R | Continuous |
| `inside_resistance_zone` | S/R | Binary |
| `inside_support_zone` | S/R | Binary |
| `resistance_rejection_risk` | S/R | Categorical |
| `resistance_broken_flag` | S/R | Binary |
| `at_52w_high_flag` | Price Memory | Binary |
| `dist_to_52w_high_pct` | Price Memory | Continuous |
| `at_round_number_flag` | Psych Levels | Binary |
| `dist_to_round_number_pct` | Psych Levels | Continuous |
| `fib_618_level`, `fib_382_level` | Fibonacci | Continuous |
| `at_fib_618_flag`, `at_fib_382_flag` | Fibonacci | Binary |
| `fib_confluence_flag` | Fibonacci | Binary |
| `fib_bounce_setup` | Fibonacci | Binary |
| `fib_ext_1272`, `fib_ext_1618` | Fibonacci | Continuous |
| `volume_declining_pullback` | Volume | Binary |
| `double_top_flag` | Patterns | Binary |
| `double_bottom_flag` | Patterns | Binary |
| `hs_pattern_flag` | Patterns | Binary |
| `inverse_hs_flag` | Patterns | Binary |
| `ascending_triangle_flag` | Patterns | Binary |
| `triangle_breakout_flag` | Patterns | Binary |
| `bull_flag_forming` | Patterns | Binary |
| `bull_flag_breakout` | Patterns | Binary |
| `rising_wedge_flag` | Patterns | Binary |
| `falling_wedge_flag` | Patterns | Binary |
| `three_pushes_high_flag` | Patterns | Binary |
| `three_pushes_low_flag` | Patterns | Binary |
| `wyckoff_phase` | Wyckoff | Categorical |
| `wyckoff_accumulation_flag` | Wyckoff | Binary |
| `wyckoff_distribution_flag` | Wyckoff | Binary |
| `nearest_demand_zone_low/high` | Supply/Demand | Continuous |
| `inside_demand_zone` | Supply/Demand | Binary |
| `inside_supply_zone` | Supply/Demand | Binary |
| `zone_test_count` | Supply/Demand | Integer |
| `liquidity_sweep_bullish` | SMC | Binary |
| `liquidity_sweep_bearish` | SMC | Binary |
| `inside_bullish_ob` | SMC | Binary |
| `open_fvg_above` | SMC | Continuous |
| `open_fvg_below` | SMC | Continuous |
| `dist_to_nearest_fvg_pct` | SMC | Continuous |
| `market_structure_bullish` | SMC | Binary |
| `bos_bullish` | SMC | Binary |
| `choch_bearish` | SMC | Binary |
| `hammer_at_support` | Candles+Level | Binary |
| `shooting_star_at_resistance` | Candles+Level | Binary |
| `bullish_pinbar_at_level` | Candles+Level | Binary |
| `bearish_pinbar_at_level` | Candles+Level | Binary |
| `evening_star_at_resistance` | Candles+Level | Binary |
| `morning_star_at_support` | Candles+Level | Binary |
| `upper_wick_resistance` | Candles | Binary |
| `momentum_continuation` | Candles | Binary |
| `volume_climax_flag` | Volume | Binary |
| `low_vol_advance_flag` | Volume | Binary |
| `pullback_volume_dryup` | Volume | Binary |
| `breakaway_gap_flag` | Gaps | Binary |
| `runaway_gap_flag` | Gaps | Binary |
| `exhaustion_gap_flag` | Gaps | Binary |
| `common_gap_flag` | Gaps | Binary |
| `gap_trap_flag` | VN-Specific | Binary |
| `post_ceiling_day1_flag` | VN-Specific | Binary |
| `post_ceiling_day3_plus_flag` | VN-Specific | Binary |
| `post_floor_day3_plus_flag` | VN-Specific | Binary |
| `t3_exit_pressure` | VN-Specific | Binary |
| `earnings_season_flag` | VN-Specific | Binary |
| `rotation_stage` | Sector | Integer (1-5) |
| `is_sector_leader` | Sector | Binary |
| `sector_leader_broke_resistance` | Sector | Binary |
| `day_of_week_effect` | Timing | Integer (0-4) |

**Total features after Part F**: ~190 features per stock per day.

---

