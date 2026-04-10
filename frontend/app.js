/* ═══════════════════════════════════════════════════════════
   VN100 Quant Signal Dashboard — app.js
   ═══════════════════════════════════════════════════════════
   Fixes in this version:
   - priceScale detection: vnstock KBS returns prices in VND
     thousands (e.g. 25 for ACB). Detected by checking if the
     last close < 500; if so all price values are × 1000.
   - Signal card prices recalculated from d.price when stored
     signal values are 0 or stale (entry_lo, target, stop_loss).
   - Vertical + horizontal crosshair dashed lines on chart hover
     via a separate overlay canvas.
   - Smooth fade + slide animation when switching tickers.
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────────────────
let UI_DATA       = null;
let currentTicker = null;
let currentTab    = 'live';
let chartData     = [];
let hoverIdx      = -1;           // crosshair position

// Chart geometry cache (set when the main chart is drawn)
const CHART = { pad: null, xOf: null, yOf: null, cW: 0, H: 0, W: 0, dpr: 1 };

// ── Formatters ─────────────────────────────────────────────
const fmt  = n  => Math.round(n).toLocaleString('vi-VN');
const fmtP = n  => (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
const fmtD = d  => {
  const dt = new Date(d);
  return dt.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric', year: '2-digit' });
};

// ── Price scale detection ───────────────────────────────────
// vnstock KBS returns prices in VND thousands for some sources.
// VN stocks range 1,000–200,000 VND; if close < 500 → ×1000.
function getPriceScale(d) {
  const p = d && d.price;
  if (!p) return 1;
  return p > 0 && p < 500 ? 1000 : 1;
}

// Apply scale to a single price value
function sp(val, scale) {
  const v = parseFloat(val) || 0;
  return v * scale;
}

// Round price to nearest sensible tick (100 or 1,000 VND)
function roundTick(price) {
  if (price < 10000)  return Math.round(price / 100)  * 100;
  if (price < 100000) return Math.round(price / 500)  * 500;
  return Math.round(price / 1000) * 1000;
}

// ── Data paths ─────────────────────────────────────────────
const DATA_PATHS = [
  '../viz/ui_data.json',         // standard: frontend/ + viz/
  'ui_data.json',                // index.html next to data
  '../data/reports/ui_data.json',
];

async function loadData() {
  for (const path of DATA_PATHS) {
    try {
      const r = await fetch(path);
      if (!r.ok) continue;
      const data = await r.json();
      document.getElementById('status-pill').className = 'status-pill status-live';
      document.getElementById('status-text').textContent =
        'LIVE · ' + new Date(data.generated_at).toLocaleTimeString('vi-VN');
      console.info('[VN100] Loaded from:', path);
      return data;
    } catch (_) { /* try next */ }
  }
  console.warn('[VN100] No data file found — using synthetic demo');
  document.getElementById('status-pill').className = 'status-pill status-static';
  document.getElementById('status-text').textContent = 'DEMO · run pipeline first';
  return buildDemoData();
}

// ── Boot ───────────────────────────────────────────────────
async function init() {
  UI_DATA = await loadData();
  const tickers = Object.keys(UI_DATA.tickers || {});
  if (!tickers.length) {
    document.getElementById('loading').classList.add('hidden');
    return;
  }
  buildTickerBar(tickers);
  buildAllSignalsTab();
  await selectTicker(tickers[0]);
  document.getElementById('loading').classList.add('hidden');
}

// ── Ticker bar ─────────────────────────────────────────────
function buildTickerBar(tickers) {
  const bar = document.getElementById('ticker-bar');
  bar.innerHTML = '';
  tickers.forEach(t => {
    const sig  = (UI_DATA.tickers[t].signal || {});
    const tier = sig.tier || 0;
    const btn  = document.createElement('button');
    btn.className = 't-btn';
    btn.id = `tb-${t}`;
    btn.innerHTML = t + (tier > 0 ? `<span class="tier-dot tier-${tier}"></span>` : '');
    btn.onclick = () => selectTicker(t);
    bar.appendChild(btn);
  });
}

// ── Ticker switch with transition ──────────────────────────
async function selectTicker(ticker) {
  if (ticker === currentTicker) return;

  // Deactivate old button
  if (currentTicker) {
    const old = document.getElementById(`tb-${currentTicker}`);
    if (old) old.classList.remove('active');
  }

  const area = document.getElementById('content-area');

  // Fade out
  area.classList.add('fading');
  await new Promise(r => setTimeout(r, 180));

  // Swap content
  currentTicker = ticker;
  const btn = document.getElementById(`tb-${ticker}`);
  if (btn) btn.classList.add('active');

  const d = UI_DATA.tickers[ticker];
  renderStats(d);
  renderSignalCard(d);
  renderFeatList(d);
  renderCharts(d);
  if (currentTab === 'backtest') renderBacktest(d);

  // Fade in
  area.classList.remove('fading');
}

// ── Stats row ──────────────────────────────────────────────
function renderStats(d) {
  const ind   = d.indicators || {};
  const scale = getPriceScale(d);
  const price = sp(d.price, scale);
  const chgP  = d.changePercent || 0;

  const grid = document.getElementById('stats-grid');
  grid.innerHTML = `
    <div class="stat-card" style="--accent-color:${chgP >= 0 ? 'var(--green)' : 'var(--red)'}">
      <div class="stat-label">${d.ticker} · ${d.sector}</div>
      <div class="stat-value">${fmt(price)} <span style="font-size:13px;color:var(--muted)">₫</span></div>
      <div class="stat-sub ${chgP >= 0 ? 'up' : 'down'}">${fmtP(chgP)} hôm nay</div>
    </div>
    <div class="stat-card" style="--accent-color:${ind.hurst > 0.55 ? 'var(--green)' : ind.hurst < 0.45 ? 'var(--red)' : 'var(--amber)'}">
      <div class="stat-label">Market Regime (Hurst)</div>
      <div class="stat-value ${ind.hurst > 0.55 ? 'up' : ind.hurst < 0.45 ? 'down' : 'amb'}">${(ind.hurst || 0.5).toFixed(3)}</div>
      <div class="stat-sub neu">${ind.regime || '—'}</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--blue)">
      <div class="stat-label">ADX (Trend Strength)</div>
      <div class="stat-value ${(ind.adx14 || 0) > 25 ? 'up' : 'neu'}">${(ind.adx14 || 0).toFixed(1)}</div>
      <div class="stat-sub neu">${(ind.adx14 || 0) > 40 ? 'Rất mạnh' : (ind.adx14 || 0) > 25 ? 'Mạnh' : 'Yếu'}</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--purple)">
      <div class="stat-label">Wyckoff Phase</div>
      <div class="stat-value" style="color:var(--purple)">${ind.wyckoffPhase || 'B'}</div>
      <div class="stat-sub neu">Uptrend Quality: ${Math.round((ind.uptrend_quality || 0) * 100)}%</div>
    </div>
  `;
}

// ── Signal card ────────────────────────────────────────────
function renderSignalCard(d) {
  const sig   = d.signal || {};
  const wrap  = document.getElementById('signal-card-wrap');
  const scale = getPriceScale(d);

  const signal = sig.signal || 'NONE';
  const tier   = sig.tier   || 0;

  if (signal === 'NONE' || !tier) {
    wrap.innerHTML = `
      <div class="card" style="padding:20px;text-align:center;">
        <div style="color:var(--muted);font-family:monospace;font-size:12px;padding:20px 0;">
          Không có tín hiệu hôm nay<br>
          <span style="font-size:10px;opacity:.5;">Run pipeline to generate signals</span>
        </div>
      </div>`;
    return;
  }

  // ── Price values with scale + recalculation fallback ─────
  // Use d.price (from features last-row) as the authoritative
  // current price; sig.current_price may be in thousands.
  const currentPrice = sp(d.price, scale);

  // Stored signal prices might be stale (0 or in thousands).
  // Accept them only if they look sensible relative to current price.
  function useOrCalc(stored, factor) {
    const v = sp(stored, scale);
    // Valid if within 50% of current price and positive
    if (v > currentPrice * 0.5 && v < currentPrice * 1.5) return v;
    return roundTick(currentPrice * factor);
  }

  const entryLo  = useOrCalc(sig.entry_lo,  0.982);
  const entryHi  = useOrCalc(sig.entry_hi,  1.008);
  const target   = useOrCalc(sig.target,    1.10);
  const stopLoss = useOrCalc(sig.stop_loss, 0.942);

  const rewardPct = ((target   - currentPrice) / currentPrice * 100);
  const riskPct   = ((currentPrice - stopLoss) / currentPrice * 100);
  const rrRatio   = riskPct > 0 ? (rewardPct / riskPct) : 0;
  const rrBarPct  = Math.min(Math.max((rrRatio - 0.5) / 2.5 * 100, 5), 95);

  const sigColor  = { BUY:'var(--green)', SELL:'var(--red)', HOLD:'var(--blue)', NONE:'var(--muted)' }[signal] || 'var(--muted)';
  const badgeClass = { BUY:'badge-buy', SELL:'badge-sell', HOLD:'badge-hold', NONE:'badge-none' }[signal] || 'badge-none';
  const sigLabel  = { BUY:'▲ MUA', SELL:'▼ BÁN', HOLD:'◆ GIỮ', NONE:'— KHÔNG CÓ' }[signal] || '—';

  const dirProb  = sig.model_dir_prob || sig.dir_prob || 0;
  const winRate  = sig.win_rate_hist  || 0;
  const nSig     = sig.n_signals      || 0;

  let dots = '';
  for (let i = 0; i < 9; i++) {
    dots += `<div class="sig-dot ${i < nSig ? 'on' : ''}"></div>`;
  }

  wrap.innerHTML = `
    <div class="signal-card" style="--sig-color:${sigColor};">
      <div class="sig-header">
        <div>
          <div class="sig-ticker">${d.ticker}</div>
          <div class="sig-name">${d.name}</div>
          <div class="sig-date">${sig.date || '—'}</div>
        </div>
        <div style="text-align:right;">
          <span class="sig-badge ${badgeClass}">${sigLabel}</span>
          <div style="font-size:10px;color:var(--muted);margin-top:6px;">${sig.tier_label || `TIER ${tier}`}</div>
        </div>
      </div>

      <div class="sig-body">
        <div style="margin-bottom:10px;">
          <span class="strat-tag">${sig.strategy_name || sig.primary_strategy || '—'}</span>
          <span class="hold-tag">⏱ T+${sig.hold_days || '?'}</span>
        </div>

        <div class="price-2col">
          <div class="price-box">
            <div class="lb">Giá hiện tại</div>
            <div class="vl p-now">${fmt(currentPrice)} ₫</div>
          </div>
          <div class="price-box">
            <div class="lb">Vùng mua vào</div>
            <div class="vl p-entry" style="font-size:12px;">${fmt(entryLo)} – ${fmt(entryHi)}</div>
          </div>
          <div class="price-box">
            <div class="lb">Mục tiêu</div>
            <div class="vl p-target">+${rewardPct.toFixed(1)}% · ${fmt(target)}</div>
          </div>
          <div class="price-box">
            <div class="lb">Cắt lỗ</div>
            <div class="vl p-stop">−${riskPct.toFixed(1)}% · ${fmt(stopLoss)}</div>
          </div>
        </div>

        <div class="rr-wrap">
          <div class="rr-labels">
            <span>Rủi ro: −${riskPct.toFixed(1)}%</span>
            <span style="color:var(--text)">R:R = ${rrRatio.toFixed(2)}:1</span>
            <span>LN: +${rewardPct.toFixed(1)}%</span>
          </div>
          <div class="rr-track"><div class="rr-fill" style="width:${rrBarPct}%"></div></div>
        </div>

        <div class="ind-3col">
          <div class="ind-box">
            <div class="ib-l">RSI 14</div>
            <div class="ib-v ${(sig.rsi_14 || 50) < 30 ? 'up' : (sig.rsi_14 || 50) > 70 ? 'down' : 'neu'}">${(sig.rsi_14 || 0).toFixed(1)}</div>
          </div>
          <div class="ind-box">
            <div class="ib-l">ADX 14</div>
            <div class="ib-v ${(sig.adx_14 || 0) > 25 ? 'up' : 'neu'}">${(sig.adx_14 || 0).toFixed(1)}</div>
          </div>
          <div class="ind-box">
            <div class="ib-l">KL/TB</div>
            <div class="ib-v ${(sig.rel_volume || 1) > 2 ? 'up' : 'neu'}">${(sig.rel_volume || 0).toFixed(2)}×</div>
          </div>
        </div>

        <div class="conf-row">
          <div class="conf-cell">
            <div class="conf-label">Xác suất</div>
            <div class="conf-val">${dirProb.toFixed(1)}%</div>
          </div>
          <div class="conf-cell">
            <div class="conf-label">Tín hiệu</div>
            <div class="conf-val">${nSig}/9</div>
            <div class="dots-row">${dots}</div>
          </div>
          <div class="conf-cell">
            <div class="conf-label">Win rate</div>
            <div class="conf-val">${winRate.toFixed(1)}%</div>
          </div>
        </div>

        <div class="cancel-box">
          <strong>🛑 Điều kiện hủy tín hiệu</strong>
          Thoát ngay nếu giá đóng cửa dưới <strong>${fmt(stopLoss)} ₫</strong> ·
          Ngoại bán ròng mạnh + khối lượng sụt.
        </div>
      </div>
    </div>`;
}

// ── Feature / Macro lists ──────────────────────────────────
function renderFeatList(d) {
  const ind = d.indicators || {};
  const row = (lbl, val, cls = '') =>
    `<div class="feat-row"><span class="feat-label">${lbl}</span><span class="feat-value ${cls}">${val}</span></div>`;

  document.getElementById('feat-list').innerHTML =
    row('GARCH σ (annualised)', `${(ind.garchSigma || 0).toFixed(1)}%`, 'neu') +
    row('VSA Ratio',            `${(ind.vsaRatio    || 0).toFixed(3)}`,  'neu') +
    row('VWAP Deviation',       `${(ind.vwapDeviation || 0) >= 0 ? '+' : ''}${(ind.vwapDeviation || 0).toFixed(2)}%`,
        (ind.vwapDeviation || 0) > 0 ? 'up' : 'down') +
    row('Ceiling Demand',       `${ind.ceilingDemand || 0}`, 'neu') +
    row('Bull Trap Flag',       ind.bullTrapFlag ? 'DETECTED' : 'CLEAR', ind.bullTrapFlag ? 'down' : 'up') +
    row('Bear Trap Flag',       ind.bearTrapFlag ? 'DETECTED' : 'CLEAR', ind.bearTrapFlag ? 'up'   : 'down') +
    row('FOMO Signal',          ind.fomoSignal   ? 'ACTIVE'   : 'CLEAR', ind.fomoSignal   ? 'down' : 'up') +
    row('T+2.5 Risk',           `${(ind.t25Risk || 0).toFixed(1)}%`,     (ind.t25Risk || 0) > 6 ? 'down' : 'up');

  document.getElementById('macro-list').innerHTML =
    row('Breadth > 50-SMA',    `${(ind.breadthPctAbove50 || 50).toFixed(1)}%`,  (ind.breadthPctAbove50 || 50) > 50 ? 'up' : 'down') +
    row('A/D Ratio 10d',       `${(ind.adRatio10d || 1).toFixed(2)}`,           (ind.adRatio10d || 1) > 1 ? 'up' : 'down') +
    row('Interbank Rate',      `${(ind.interbankRate || 3.5).toFixed(1)}%`,      (ind.interbankRate || 3.5) < 4 ? 'up' : 'down') +
    row('USD/VND Trend 10d',   `${(ind.usdVndTrend || 0) >= 0 ? '+' : ''}${(ind.usdVndTrend || 0).toFixed(2)}%`,
        (ind.usdVndTrend || 0) < 0 ? 'up' : 'down') +
    row('VN-Index MA Score',   `${(ind.vnIndexScore || 0.5).toFixed(2)}`,        (ind.vnIndexScore || 0.5) > 0.5 ? 'up' : 'down');
}

// ── Charts ─────────────────────────────────────────────────
function renderCharts(d) {
  const scale = getPriceScale(d);
  // Scale all chart prices
  chartData = (d.chart || []).map(r => ({
    ...r,
    close:       r.close      * scale,
    open:        r.open       * scale,
    high:        r.high       * scale,
    low:         r.low        * scale,
    sma20:       (r.sma20     || 0) * scale,
    sma50:       (r.sma50     || 0) * scale,
    sma200:      (r.sma200    || 0) * scale,
    bb_upper:    (r.bb_upper  || 0) * scale,
    bb_lower:    (r.bb_lower  || 0) * scale,
    kalman:      (r.kalman    || 0) * scale,
    upperBound:  (r.upperBound|| 0) * scale,
    lowerBound:  (r.lowerBound|| 0) * scale,
    sar:         (r.sar       || 0) * scale,
  }));
  redrawChart();
  drawVolChart();
  drawRsiChart();
  syncOverlaySize();
}

function redrawChart() {
  const canvas  = document.getElementById('price-chart');
  const showSma = document.getElementById('toggle-sma')?.checked;
  const showBB  = document.getElementById('toggle-bb')?.checked;
  const showSar = document.getElementById('toggle-sar')?.checked;
  drawPriceChart(canvas, chartData, { showSma, showBB, showSar });
  drawCrosshair(hoverIdx);
}

// ── Main price chart ───────────────────────────────────────
function drawPriceChart(canvas, data, opts = {}) {
  if (!data || !data.length) return;
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.offsetWidth || 600;
  const H   = 320;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = { top: 20, right: 12, bottom: 30, left: 68 };
  const cW  = W - pad.left - pad.right;
  const cH  = H - pad.top  - pad.bottom;

  // Price range
  const allP = data.flatMap(r => [
    r.close, r.upperBound, r.lowerBound,
    opts.showBB ? r.bb_upper : 0, opts.showBB ? r.bb_lower : 0,
  ].filter(Boolean));
  let pMin = Math.min(...allP) * 0.998;
  let pMax = Math.max(...allP) * 1.002;
  if (pMax === pMin) { pMin *= 0.99; pMax *= 1.01; }

  const xOf = i => pad.left + (i / (data.length - 1)) * cW;
  const yOf = p => pad.top  + (1 - (p - pMin) / (pMax - pMin)) * cH;

  // Cache geometry for crosshair
  CHART.pad = pad; CHART.xOf = xOf; CHART.yOf = yOf;
  CHART.cW = cW; CHART.H = H; CHART.W = W; CHART.dpr = dpr;

  // Grid lines + Y labels
  ctx.strokeStyle = '#1a2340'; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.top + (g / 4) * cH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const p = pMax - (g / 4) * (pMax - pMin);
    ctx.fillStyle   = '#5a6b8a';
    ctx.font        = '10px IBM Plex Mono, monospace';
    ctx.textAlign   = 'right';
    ctx.fillText(fmt(p), pad.left - 4, y + 4);
  }

  // X axis date labels
  const step = Math.max(1, Math.floor(data.length / 6));
  ctx.fillStyle = '#5a6b8a'; ctx.font = '9px IBM Plex Mono, monospace'; ctx.textAlign = 'center';
  data.forEach((r, i) => {
    if (i % step === 0) ctx.fillText(fmtD(r.date), xOf(i), H - pad.bottom + 14);
  });

  // GARCH bounds fill
  ctx.beginPath();
  data.forEach((r, i) => {
    const y = yOf(r.upperBound || r.close);
    i === 0 ? ctx.moveTo(xOf(i), y) : ctx.lineTo(xOf(i), y);
  });
  [...data].reverse().forEach((r, i) => {
    ctx.lineTo(xOf(data.length - 1 - i), yOf(r.lowerBound || r.close));
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(77,159,255,0.05)';
  ctx.fill();

  // Bollinger Bands
  if (opts.showBB) {
    ['bb_upper', 'bb_lower'].forEach(k => {
      ctx.beginPath(); ctx.strokeStyle = 'rgba(167,139,250,0.35)';
      ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      let first = true;
      data.forEach((r, i) => {
        const v = r[k]; if (!v) return;
        first ? (ctx.moveTo(xOf(i), yOf(v)), first = false) : ctx.lineTo(xOf(i), yOf(v));
      });
      ctx.stroke(); ctx.setLineDash([]);
    });
  }

  // GARCH bound outlines
  ['upperBound', 'lowerBound'].forEach(k => {
    ctx.beginPath(); ctx.strokeStyle = 'rgba(74,85,104,0.4)';
    ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    let first = true;
    data.forEach((r, i) => {
      const v = r[k]; if (!v) return;
      first ? (ctx.moveTo(xOf(i), yOf(v)), first = false) : ctx.lineTo(xOf(i), yOf(v));
    });
    ctx.stroke(); ctx.setLineDash([]);
  });

  // SMA lines
  if (opts.showSma) {
    [['sma20', '#ffb340', 1.2], ['sma50', 'rgba(34,211,238,.65)', 1]].forEach(([k, col, lw]) => {
      ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = lw; let first = true;
      data.forEach((r, i) => {
        const v = r[k]; if (!v) return;
        first ? (ctx.moveTo(xOf(i), yOf(v)), first = false) : ctx.lineTo(xOf(i), yOf(v));
      });
      ctx.stroke();
    });
  }

  // Kalman
  ctx.beginPath(); ctx.strokeStyle = 'rgba(167,139,250,.75)'; ctx.lineWidth = 1.5; let fk = true;
  data.forEach((r, i) => {
    const v = r.kalman; if (!v) return;
    fk ? (ctx.moveTo(xOf(i), yOf(v)), fk = false) : ctx.lineTo(xOf(i), yOf(v));
  });
  ctx.stroke();

  // Close price line
  ctx.beginPath(); ctx.strokeStyle = '#4d9fff'; ctx.lineWidth = 2; let fp = true;
  data.forEach((r, i) => {
    fp ? (ctx.moveTo(xOf(i), yOf(r.close)), fp = false) : ctx.lineTo(xOf(i), yOf(r.close));
  });
  ctx.stroke();

  // SAR dots
  if (opts.showSar) {
    data.forEach((r, i) => {
      if (!r.sar) return;
      ctx.beginPath();
      ctx.arc(xOf(i), yOf(r.sar), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = r.sarBullish ? 'var(--green)' : 'var(--red)';
      ctx.fill();
    });
  }
}

// ── Overlay canvas: crosshair ──────────────────────────────
function syncOverlaySize() {
  const base    = document.getElementById('price-chart');
  const overlay = document.getElementById('price-overlay');
  if (!base || !overlay) return;
  overlay.style.width  = base.style.width  || '100%';
  overlay.style.height = base.offsetHeight + 'px';
  overlay.width  = base.width;
  overlay.height = base.height;
}

function drawCrosshair(idx) {
  const overlay = document.getElementById('price-overlay');
  if (!overlay || !CHART.pad || idx < 0 || !chartData.length) {
    if (overlay) {
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }
    return;
  }

  const dpr = CHART.dpr;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const { pad, xOf, yOf, H, W } = CHART;
  const r = chartData[idx];
  const x = xOf(idx);
  const y = yOf(r.close);

  // Vertical dashed line
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(221,229,245,0.25)';
  ctx.lineWidth   = 1;
  ctx.moveTo(x, pad.top);
  ctx.lineTo(x, H - pad.bottom);
  ctx.stroke();

  // Horizontal dashed line
  ctx.beginPath();
  ctx.moveTo(pad.left, y);
  ctx.lineTo(W - pad.right, y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Dot at close
  ctx.beginPath();
  ctx.arc(x, y, 3.5, 0, Math.PI * 2);
  ctx.fillStyle   = '#4d9fff';
  ctx.strokeStyle = '#060810';
  ctx.lineWidth   = 1.5;
  ctx.fill();
  ctx.stroke();

  // Price label on Y axis
  const priceStr = fmt(r.close) + ' ₫';
  ctx.font      = 'bold 10px IBM Plex Mono, monospace';
  const tw      = ctx.measureText(priceStr).width;
  ctx.fillStyle = '#4d9fff';
  ctx.fillRect(pad.left - tw - 10, y - 9, tw + 8, 18);
  ctx.fillStyle   = '#060810';
  ctx.textAlign   = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(priceStr, pad.left - 4, y);

  ctx.restore();
}

// ── Mouse events on chart wrap ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const wrap = document.getElementById('price-chart-wrap');
  if (!wrap) return;

  wrap.addEventListener('mousemove', e => {
    if (!chartData.length || !CHART.pad) return;
    const rect  = document.getElementById('price-chart').getBoundingClientRect();
    const mx    = e.clientX - rect.left;
    const rawIdx = (mx - CHART.pad.left) / CHART.cW * (chartData.length - 1);
    const idx   = Math.max(0, Math.min(chartData.length - 1, Math.round(rawIdx)));
    hoverIdx    = idx;
    drawCrosshair(idx);

    // Tooltip
    const r = chartData[idx];
    const prev = chartData[idx - 1];
    const chg  = prev ? ((r.close - prev.close) / prev.close * 100) : 0;
    document.getElementById('tt-date').textContent  = r.date;
    document.getElementById('tt-close').textContent = fmt(r.close) + ' ₫';
    document.getElementById('tt-chg').textContent   = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
    document.getElementById('tt-chg').style.color   = chg >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('tt-vol').textContent   = (r.volume / 1e6).toFixed(2) + 'M';
    document.getElementById('tt-rsi').textContent   = (r.rsi || 0).toFixed(1);
    document.getElementById('tt-adx').textContent   = (r.adx || 0).toFixed(1);

    const tt = document.getElementById('chart-tooltip');
    tt.style.display = 'block';
    tt.style.left    = (e.clientX + 14) + 'px';
    tt.style.top     = (e.clientY - 20) + 'px';
  });

  wrap.addEventListener('mouseleave', () => {
    hoverIdx = -1;
    drawCrosshair(-1);
    document.getElementById('chart-tooltip').style.display = 'none';
  });
});

// ── Volume chart ───────────────────────────────────────────
function drawVolChart() {
  const canvas = document.getElementById('vol-chart');
  const data   = chartData;
  if (!data.length) return;

  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.offsetWidth || 600;
  const H   = 100;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad  = { top: 6, right: 12, bottom: 20, left: 68 };
  const cW   = W - pad.left - pad.right;
  const cH   = H - pad.top  - pad.bottom;
  const maxV = Math.max(...data.map(r => r.volume || 0));
  const adv20 = data.reduce((s, r) => s + (r.volume || 0), 0) / data.length;
  const barW  = Math.max(1, cW / data.length - 0.5);

  data.forEach((r, i) => {
    const x   = pad.left + (i / (data.length - 1)) * cW;
    const h   = ((r.volume || 0) / maxV) * cH;
    const y   = pad.top + cH - h;
    ctx.fillStyle = (r.relVolume || 1) > 2 ? 'rgba(0,229,160,.55)' : 'rgba(74,85,104,.45)';
    ctx.fillRect(x - barW / 2, y, barW, h);
  });

  // ADV line
  const avgH = (adv20 / maxV) * cH;
  ctx.beginPath(); ctx.strokeStyle = 'rgba(255,179,64,.5)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.moveTo(pad.left, pad.top + cH - avgH);
  ctx.lineTo(W - pad.right, pad.top + cH - avgH);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,179,64,.5)';
  ctx.font = '9px IBM Plex Mono, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('ADV', pad.left - 2, pad.top + cH - avgH + 4);
}

// ── RSI + MACD chart ───────────────────────────────────────
function drawRsiChart() {
  const canvas = document.getElementById('rsi-chart');
  const data   = chartData;
  if (!data.length) return;

  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.offsetWidth || 600;
  const H   = 80;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = { top: 6, right: 12, bottom: 4, left: 68 };
  const cW  = W - pad.left - pad.right;
  const cH  = H - pad.top  - pad.bottom;
  const xOf = i => pad.left + (i / (data.length - 1)) * cW;
  const rsiY = v => pad.top + (1 - v / 100) * cH;

  // Background zones
  ctx.fillStyle = 'rgba(0,229,160,.05)';
  ctx.fillRect(pad.left, rsiY(30), cW, rsiY(0) - rsiY(30));
  ctx.fillStyle = 'rgba(255,77,106,.05)';
  ctx.fillRect(pad.left, rsiY(100), cW, rsiY(70) - rsiY(100));

  // Grid lines
  [30, 50, 70].forEach(v => {
    ctx.beginPath(); ctx.strokeStyle = '#1a2340'; ctx.lineWidth = 1;
    ctx.moveTo(pad.left, rsiY(v)); ctx.lineTo(W - pad.right, rsiY(v)); ctx.stroke();
    ctx.fillStyle = '#5a6b8a'; ctx.font = '9px IBM Plex Mono, monospace'; ctx.textAlign = 'right';
    ctx.fillText(v, pad.left - 4, rsiY(v) + 3);
  });

  // MACD histogram
  const macds = data.map(r => r.macd || 0);
  const mMax  = Math.max(...macds.map(Math.abs)) || 1;
  data.forEach((r, i) => {
    const v  = (r.macd || 0) / mMax * 0.4;
    const bH = Math.abs(v) * cH;
    const bY = v >= 0 ? rsiY(50 + Math.abs(v) * 50) : rsiY(50);
    ctx.fillStyle = v >= 0 ? 'rgba(0,229,160,.35)' : 'rgba(255,77,106,.35)';
    ctx.fillRect(xOf(i) - 1, bY, 2, bH);
  });

  // RSI line
  ctx.beginPath(); ctx.strokeStyle = '#ffb340'; ctx.lineWidth = 1.5; let first = true;
  data.forEach((r, i) => {
    const v = r.rsi;
    if (v == null) return;
    first ? (ctx.moveTo(xOf(i), rsiY(v)), first = false) : ctx.lineTo(xOf(i), rsiY(v));
  });
  ctx.stroke();

  // RSI badge
  const last = data[data.length - 1];
  const rsiEl = document.getElementById('rsi-value');
  if (rsiEl && last) {
    const rsiVal = (last.rsi || 50);
    rsiEl.textContent = `RSI ${rsiVal.toFixed(1)}`;
    rsiEl.style.color = rsiVal < 30 ? 'var(--green)' : rsiVal > 70 ? 'var(--red)' : 'var(--amber)';
  }
}

// ── Backtest tab ───────────────────────────────────────────
function renderBacktest(d) {
  const bt     = d.backtest  || {};
  const m      = d.metrics   || {};
  const equity = bt.equity   || [];
  const trades = bt.trades   || [];
  const scale  = getPriceScale(d);

  // Stat cards
  const btGrid = document.getElementById('bt-stats-grid');
  const c = v => v >= 0 ? 'up' : 'down';
  btGrid.innerHTML = `
    <div class="stat-card" style="--accent-color:${(m.ann_return || 0) >= 0 ? 'var(--green)' : 'var(--red)'}">
      <div class="stat-label">Annual Return (CAGR)</div>
      <div class="stat-value ${c(m.ann_return || 0)}">${((m.ann_return || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--blue)">
      <div class="stat-label">Sharpe Ratio</div>
      <div class="stat-value ${(m.sharpe || 0) >= 1 ? 'up' : 'down'}">${(m.sharpe || 0).toFixed(2)}</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--amber)">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value">${((m.win_rate || 0) * 100).toFixed(1)}%</div>
      <div class="stat-sub neu">${m.n_trades || 0} giao dịch</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--purple)">
      <div class="stat-label">Profit Factor</div>
      <div class="stat-value ${(m.profit_factor || 0) >= 1.5 ? 'up' : 'down'}">${(m.profit_factor || 0).toFixed(2)}</div>
      <div class="stat-sub ${c(m.max_drawdown || 0)}">DD: ${((m.max_drawdown || 0) * 100).toFixed(1)}%</div>
    </div>`;

  // Equity curve
  if (equity.length > 1) {
    const cvs = document.getElementById('bt-equity-chart');
    const dpr = window.devicePixelRatio || 1;
    const W   = cvs.offsetWidth || 800;
    const H   = 280;
    cvs.width  = W * dpr;
    cvs.height = H * dpr;
    const ctx  = cvs.getContext('2d');
    ctx.scale(dpr, dpr);
    const pad  = { top: 20, right: 16, bottom: 30, left: 80 };
    const cW   = W - pad.left - pad.right;
    const cH   = H - pad.top  - pad.bottom;
    const vals = equity.map(e => e.equity);
    const eMin = Math.min(...vals);
    const eMax = Math.max(...vals);
    const xOf  = i => pad.left + (i / (equity.length - 1)) * cW;
    const yOf  = v => pad.top  + (1 - (v - eMin) / (eMax - eMin || 1)) * cH;

    for (let g = 0; g <= 4; g++) {
      const y = pad.top + (g / 4) * cH;
      ctx.strokeStyle = '#1a2340'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
      const v = eMax - (g / 4) * (eMax - eMin);
      ctx.fillStyle = '#5a6b8a'; ctx.font = '9px IBM Plex Mono, monospace'; ctx.textAlign = 'right';
      ctx.fillText((v / 1e6).toFixed(0) + 'M', pad.left - 4, y + 3);
    }

    // Fill
    ctx.beginPath();
    equity.forEach((e, i) => i === 0 ? ctx.moveTo(xOf(i), yOf(e.equity)) : ctx.lineTo(xOf(i), yOf(e.equity)));
    ctx.lineTo(xOf(equity.length - 1), pad.top + cH);
    ctx.lineTo(xOf(0), pad.top + cH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(139,92,246,.1)';
    ctx.fill();

    // Line
    ctx.beginPath(); ctx.strokeStyle = '#8b5cf6'; ctx.lineWidth = 2;
    equity.forEach((e, i) => i === 0 ? ctx.moveTo(xOf(i), yOf(e.equity)) : ctx.lineTo(xOf(i), yOf(e.equity)));
    ctx.stroke();

    // Starting capital reference line
    const startY = yOf(100_000_000);
    ctx.beginPath(); ctx.strokeStyle = 'rgba(255,179,64,.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.moveTo(pad.left, startY); ctx.lineTo(W - pad.right, startY);
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Trade table
  const tbody = document.getElementById('trade-tbody');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px;">Chưa có giao dịch</td></tr>';
    return;
  }
  tbody.innerHTML = trades.slice(-40).map(t => {
    const pnl    = (t.pnl_pct || 0) * 100;
    const rowCls = pnl >= 0 ? 'win-row' : 'loss-row';
    return `<tr class="${rowCls}">
      <td>${(t.entry_date || '').slice(0, 10)}</td>
      <td>${(t.exit_date  || '').slice(0, 10)}</td>
      <td>${fmt((t.entry_price || 0) * scale)}</td>
      <td>${fmt((t.exit_price  || 0) * scale)}</td>
      <td>${(pnl >= 0 ? '+' : '') + pnl.toFixed(2)}%</td>
      <td style="color:var(--purple);font-size:10px;">${t.strategy || '—'}</td>
      <td style="color:var(--amber)">T${t.tier || '?'}</td>
      <td style="color:var(--muted);font-size:10px;">${t.reason || '—'}</td>
    </tr>`;
  }).join('');
}

// ── All signals tab ────────────────────────────────────────
function buildAllSignalsTab() {
  const wrap    = document.getElementById('all-signals-wrap');
  const summary = (UI_DATA.daily_summary || []).slice(0, 24);

  if (!summary.length) {
    wrap.innerHTML = '<p style="color:var(--muted);padding:20px;font-family:monospace;font-size:12px;">No active signals today. Run pipeline to generate signals.</p>';
    return;
  }

  wrap.innerHTML = '';
  summary.forEach(s => {
    const d = UI_DATA.tickers[s.ticker];
    if (!d) return;
    const sig   = d.signal || {};
    const scale = getPriceScale(d);
    const tier  = s.tier || sig.tier || 0;
    const col   = tier === 1 ? 'var(--green)' : tier === 2 ? 'var(--amber)' : 'var(--blue)';

    const card = document.createElement('div');
    card.className = 'signal-card';
    card.style.setProperty('--sig-color', col);
    card.style.cursor = 'pointer';
    card.innerHTML = `
      <div class="sig-header">
        <div>
          <div class="sig-ticker">${s.ticker}</div>
          <div class="sig-name">${d.name}</div>
        </div>
        <span class="sig-badge badge-buy">▲ MUA</span>
      </div>
      <div class="sig-body" style="padding:10px 14px;">
        <span class="strat-tag">${sig.strategy_name || s.strat || '—'}</span>
        <span class="hold-tag" style="margin-top:4px;">T+${sig.hold_days || '?'}</span>
        <div style="display:flex;justify-content:space-between;margin-top:10px;">
          <div>
            <div style="font-size:9px;color:var(--muted);">WIN RATE</div>
            <div style="font-family:monospace;color:var(--green)">${(sig.win_rate_hist || 0).toFixed(1)}%</div>
          </div>
          <div>
            <div style="font-size:9px;color:var(--muted);">PROB</div>
            <div style="font-family:monospace;color:var(--blue)">${(s.prob || 0).toFixed(1)}%</div>
          </div>
          <div>
            <div style="font-size:9px;color:var(--muted);">TIER</div>
            <div style="font-family:monospace;color:${col}">TIER ${tier}</div>
          </div>
        </div>
      </div>`;
    card.onclick = () => { switchTab('live'); selectTicker(s.ticker); };
    wrap.appendChild(card);
  });
}

// ── Tab switching ──────────────────────────────────────────
function switchTab(name) {
  currentTab = name;
  ['live', 'backtest', 'signals'].forEach(t => {
    document.getElementById(`tab-${t}`).style.display = t === name ? 'block' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['live', 'backtest', 'signals'][i] === name);
  });
  if (name === 'backtest' && currentTicker && UI_DATA) {
    const d = UI_DATA.tickers[currentTicker];
    if (d) setTimeout(() => renderBacktest(d), 50);
  }
}

// ── Resize handler ─────────────────────────────────────────
window.addEventListener('resize', () => {
  if (!currentTicker || !UI_DATA) return;
  const d = UI_DATA.tickers[currentTicker];
  if (d) {
    syncOverlaySize();
    redrawChart();
    drawVolChart();
    drawRsiChart();
    if (currentTab === 'backtest') renderBacktest(d);
  }
});

// ═══════════════════════════════════════════════════════════
// DEMO DATA (offline mode — used when ui_data.json not found)
// ═══════════════════════════════════════════════════════════
function buildDemoData() {
  const TICKERS = ['HPG','VCB','VHM','VNM','TCB','FPT','MWG','SSI','GAS','MSN'];
  const META = {
    HPG:{ name:'Tập đoàn Hòa Phát',  sector:'Steel',       price:26400 },
    VCB:{ name:'Vietcombank',          sector:'Banking',     price:85000 },
    VHM:{ name:'Vinhomes',             sector:'Real Estate', price:42000 },
    VNM:{ name:'Vinamilk',             sector:'Consumer',    price:68000 },
    TCB:{ name:'Techcombank',          sector:'Banking',     price:21000 },
    FPT:{ name:'FPT Corporation',      sector:'Technology',  price:95000 },
    MWG:{ name:'Mobile World',         sector:'Retail',      price:48000 },
    SSI:{ name:'SSI Securities',       sector:'Securities',  price:32000 },
    GAS:{ name:'PetroVN Gas',          sector:'Energy',      price:78000 },
    MSN:{ name:'Masan Group',          sector:'Consumer',    price:71000 },
  };
  const STRATS = [
    { key:'RSI_OVERSOLD',      name:'RSI Quá Bán',         holdDays:60,  wr:74.3 },
    { key:'VOLUME_EXPLOSION',  name:'Bùng Nổ Khối Lượng',  holdDays:180, wr:61.4 },
    { key:'DMI_WAVE',          name:'Lướt Sóng DMI',        holdDays:10,  wr:70.0 },
    { key:'SAR_MACD',          name:'SAR × MACD',           holdDays:20,  wr:70.6 },
    { key:'PRICE_DOWN_15_MA20',name:'Giá −15% vs MA20',     holdDays:180, wr:100  },
    { key:'UPTREND',           name:'Uptrend',              holdDays:180, wr:59.3 },
  ];

  function makeSeries(baseP, n = 120) {
    const rows = []; let p = baseP;
    const now = new Date();
    for (let i = n; i >= 0; i--) {
      const d = new Date(now); d.setDate(d.getDate() - i);
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      p = Math.max(p * (1 + (Math.random() - 0.48) * 0.015), baseP * 0.5);
      const vol = Math.floor(Math.exp(14.5 + Math.random() * 0.8));
      const h = p * (1 + Math.random() * 0.008), l = p * (1 - Math.random() * 0.008);
      rows.push({
        date: d.toISOString().slice(0, 10),
        open: Math.round(p * (1 + Math.random() * 0.005)), high: Math.round(h),
        low: Math.round(l), close: Math.round(p), volume: vol,
        sma20: Math.round(p * 0.995), sma50: Math.round(p * 0.985),
        bb_upper: Math.round(p * 1.02), bb_lower: Math.round(p * 0.98),
        kalman: Math.round(p * 0.999), upperBound: Math.round(p * 1.03),
        lowerBound: Math.round(p * 0.97),
        foreignFlow: Math.round((Math.random() - 0.45) * vol * p * 0.001),
        rsi: 30 + Math.random() * 40, adx: 15 + Math.random() * 35,
        macd: (Math.random() - 0.5) * 0.003 * p,
        relVolume: 0.5 + Math.random() * 3,
        sar: Math.round(p * 0.97), sarBullish: Math.random() > 0.4 ? 1 : 0,
        hurst: 0.4 + Math.random() * 0.2,
      });
    }
    return rows;
  }

  const output = { generated_at: new Date().toISOString(), tickers: {}, daily_summary: [] };

  TICKERS.forEach((t, idx) => {
    const m     = META[t];
    const chart = makeSeries(m.price);
    const last  = chart[chart.length - 1];
    const prev  = chart[chart.length - 2] || last;
    const strat = STRATS[idx % STRATS.length];
    const tier  = 1 + (idx % 3);
    const nSig  = tier === 1 ? 3 : tier === 2 ? 2 : 1;
    const prob  = strat.wr + (nSig - 1) * 5;

    const price     = last.close;
    const stopLoss  = roundTick(price * 0.942);
    const target    = roundTick(price * 1.10);
    const entryLo   = roundTick(price * 0.982);
    const entryHi   = roundTick(price * 1.008);
    const rewardPct = (target - price) / price * 100;
    const riskPct   = (price - stopLoss) / price * 100;

    output.tickers[t] = {
      ticker: t, name: m.name, sector: m.sector,
      price, change: last.close - prev.close,
      changePercent: (last.close - prev.close) / prev.close * 100,
      chart,
      signal: {
        ticker: t, signal: 'BUY', tier,
        strategy_name: strat.name, primary_strategy: strat.key,
        n_signals: nSig, hold_days: strat.holdDays,
        date: last.date, win_rate_hist: strat.wr, dir_prob: prob, model_dir_prob: prob,
        rsi_14: last.rsi, adx_14: last.adx, rel_volume: last.relVolume,
        current_price: price, entry_lo: entryLo, entry_hi: entryHi,
        target, stop_loss: stopLoss,
        reward_pct: rewardPct, risk_pct: riskPct,
        rr_ratio: rewardPct / riskPct,
        tier_label: tier === 1 ? 'TIER 1 — ĐỘ TIN CẬY CAO NHẤT' : tier === 2 ? 'TIER 2 — ĐỘ TIN CẬY CAO' : 'TIER 3 — CHỜ XÁC NHẬN',
      },
      indicators: {
        hurst: 0.4 + Math.random() * 0.3, regime: 'Trending',
        adx14: last.adx, rsi14: last.rsi, garchSigma: 15 + Math.random() * 10,
        vsaRatio: Math.random() * 0.4, vwapDeviation: (Math.random() - 0.5) * 3,
        uptrend_quality: 0.4 + Math.random() * 0.6, wyckoffPhase: ['B','C','D','E'][idx % 4],
        breadthPctAbove50: 40 + Math.random() * 30, adRatio10d: 0.8 + Math.random() * 0.8,
        interbankRate: 3 + Math.random() * 2, usdVndTrend: (Math.random() - 0.5) * 2,
        vnIndexScore: 0.3 + Math.random() * 0.7,
        bullTrapFlag: false, bearTrapFlag: false, fomoSignal: false,
        ceilingDemand: 0, t25Risk: 2 + Math.random() * 6,
      },
      metrics: {
        n_trades: 5 + Math.floor(Math.random() * 25),
        win_rate: 0.5 + Math.random() * 0.3,
        ann_return: -0.05 + Math.random() * 0.4,
        max_drawdown: -(0.05 + Math.random() * 0.2),
        sharpe: 0.5 + Math.random() * 2,
        profit_factor: 1.2 + Math.random() * 1.5,
      },
      backtest: {
        equity: (() => {
          let cap = 100_000_000; const eq = [];
          chart.forEach((r, i) => {
            if (i % 3 === 0) { cap *= 1 + (Math.random() - 0.45) * 0.02; eq.push({ date: r.date, equity: Math.round(cap) }); }
          });
          return eq;
        })(),
        trades: Array.from({ length: 8 }, (_, i) => {
          const ei = Math.floor(Math.random() * (chart.length - 15));
          const ep = chart[ei].close;
          const xp = ep * (1 + (Math.random() > 0.65 ? 1 : -1) * Math.random() * 0.08);
          return {
            entry_date: chart[ei].date, exit_date: chart[Math.min(ei + 5 + Math.floor(Math.random() * 10), chart.length - 1)].date,
            entry_price: ep, exit_price: Math.round(xp),
            pnl_pct: (xp - ep) / ep, strategy: strat.key, tier,
            reason: Math.random() > 0.7 ? 'stop_loss' : 'hold_period',
          };
        }),
      },
    };

    if (tier <= 2) output.daily_summary.push({ ticker: t, tier, strat: strat.key, date: last.date, prob });
  });

  return output;
}

// ── Start ──────────────────────────────────────────────────
init();