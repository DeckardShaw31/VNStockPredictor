/* ═══════════════════════════════════════════════════════════
   VN100 Quant Signal Dashboard — app.js v4
   All fixes consolidated in one file.
   ═══════════════════════════════════════════════════════════ */
'use strict';

/* ── STATE ──────────────────────────────────────────────── */
let UI_DATA=null,currentTicker=null,currentTab='live';
let chartData=[],hoverIdx=-1;
let activeFilter='ALL';

/* ── UTILS ──────────────────────────────────────────────── */
const fmt = n => (n==null||isNaN(n))?'—':Math.round(n).toLocaleString('vi-VN');
const fmtP = n => (n>=0?'+':'')+n.toFixed(2)+'%';
const fmtD = d => new Date(d).toLocaleDateString('vi-VN',{day:'numeric',month:'short',year:'2-digit'});
const lerp = (a,b,t) => a+(b-a)*t;
const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
const easeOut = t => 1-Math.pow(1-t,3);
const raf2 = fn => requestAnimationFrame(()=>requestAnimationFrame(fn));

function getPriceScale(d){const p=d&&(d.price||0);return p>0&&p<500?1000:1;}
function sp(val,scale){return (parseFloat(val)||0)*scale;}

function roundTick(p){
  if(p<=0)return 0;if(p<5000)return Math.round(p/50)*50;if(p<20000)return Math.round(p/100)*100;
  if(p<50000)return Math.round(p/200)*200;if(p<200000)return Math.round(p/500)*500;
  return Math.round(p/1000)*1000;
}

// Strategy target / stop percentages (matches 03_signals.py)
const TARGET_BY_STRAT={
  PRICE_DOWN_15_MA20:0.20,RSI_OVERSOLD:0.15,PRICE_DOWN_15_20D:0.08,
  DMI_WAVE:0.08,SAR_MACD:0.10,BOLLINGER:0.05,VOLUME_EXPLOSION:0.18,
  UPTREND:0.15,STOCH_RSI:0.15,PULLBACK_BUY:0.10,GOLDEN_CROSS:0.15,
  SUPPORT_BOUNCE:0.08,LIQUIDITY_SWEEP:0.07,FIB_BOUNCE:0.10,
};
const STOP_BY_STRAT={
  PRICE_DOWN_15_MA20:0.05,RSI_OVERSOLD:0.05,PRICE_DOWN_15_20D:0.04,
  DMI_WAVE:0.04,SAR_MACD:0.05,BOLLINGER:0.03,VOLUME_EXPLOSION:0.06,
  UPTREND:0.06,STOCH_RSI:0.06,PULLBACK_BUY:0.05,GOLDEN_CROSS:0.06,
  SUPPORT_BOUNCE:0.04,LIQUIDITY_SWEEP:0.04,FIB_BOUNCE:0.05,
};

/* Compute signal levels from BASE price (price at signal date) */
function computeSignalPrices(sig, basePrice) {
  const tgtPct = TARGET_BY_STRAT[sig.primary_strategy] || 0.12;
  const slPct  = STOP_BY_STRAT[sig.primary_strategy]   || 0.05;
  const target   = basePrice*(1+tgtPct);
  const stopLoss = basePrice*(1-slPct);
  const entryLo  = basePrice*0.980;
  const entryHi  = basePrice*1.010;
  const rewardPct= (target   -basePrice)/basePrice*100;
  const riskPct  = (basePrice-stopLoss )/basePrice*100;
  const rrRatio  = riskPct>0?rewardPct/riskPct:0;
  return{entryLo,entryHi,target,stopLoss,rewardPct,riskPct,rrRatio};
}

/* Display probability — guard against raw decimal from model */
function _dispProb(sig){
  const raw=sig.model_dir_prob||sig.dir_prob||0;
  const v=parseFloat(raw);
  const pct=v<1?v*100:v;
  return Math.max(pct,(sig.win_rate_hist||50)*0.80).toFixed(1);
}

/* Signal state — uses base_price (signal date price) as anchor */
function getSignalState(sig, currentPrice, scale) {
  // Prefer pre-computed state from export pipeline
  if(sig.signal_state&&sig.signal_state.code) return sig.signal_state;
  if(!sig.tier||!sig.signal||sig.signal==='NONE') return null;

  const basePriceRaw = parseFloat(sig.base_price||sig.current_price||0);
  const base = sp(basePriceRaw, scale);
  if(base<=0) return null;

  const tgtPct=TARGET_BY_STRAT[sig.primary_strategy]||0.12;
  const slPct =STOP_BY_STRAT[sig.primary_strategy]  ||0.05;
  const target   = base*(1+tgtPct);
  const stopLoss = base*(1-slPct);
  const entryLo  = base*0.980;
  const entryHi  = base*1.010;
  const expDate  = sig.exit_date||'9999-12-31';
  const today    = new Date().toISOString().slice(0,10);
  const heldPct  = ((currentPrice-base)/base*100).toFixed(1);

  if(currentPrice>=target)  return{code:'TARGET_HIT', label:'🎯 Chốt lời',       cls:'state-target', action:'SELL',  held_pct:heldPct};
  if(currentPrice<=stopLoss)return{code:'STOP_HIT',   label:'🔴 Cắt lỗ ngay',    cls:'state-stop',   action:'SELL',  held_pct:heldPct};
  if(today>=expDate)        return{code:'EXPIRED',    label:'⏰ Hết hạn',         cls:'state-expired',action:'REVIEW',held_pct:heldPct};
  if(currentPrice>=entryLo*0.99&&currentPrice<=entryHi*1.01)
                            return{code:'ENTRY_ZONE', label:'🟡 Vùng mua vào',   cls:'state-entry',  action:'BUY',   held_pct:heldPct};
  if(currentPrice>entryHi)  return{code:'HOLDING',   label:'🔵 Đang giữ lệnh',  cls:'state-holding',action:'HOLD',  held_pct:heldPct};
  return                          {code:'WATCHING',   label:'👀 Chờ điểm vào',   cls:'state-watch',  action:'WATCH', held_pct:heldPct};
}

/* Today's recommended action text for a ticker */
function getTodayAction(d){
  const sig=d.signal||{};
  const scale=getPriceScale(d);
  const cp=sp(d.price,scale);
  if(!sig.tier||sig.signal==='NONE') return {action:'WATCH',label:'— Không có tín hiệu',cls:'state-watch'};
  const state=getSignalState(sig,cp,scale);
  if(!state) return {action:'WATCH',label:'— Chờ',cls:'state-watch'};
  return state;
}

/* ── DATA LOADING ───────────────────────────────────────── */
async function loadData(){
  for(const p of['../viz/ui_data.json','ui_data.json','../data/reports/ui_data.json']){
    try{const r=await fetch(p);if(!r.ok)continue;
      const data=await r.json();
      document.getElementById('status-pill').className='status-pill status-live';
      document.getElementById('status-text').textContent='LIVE · '+new Date(data.generated_at).toLocaleTimeString('vi-VN');
      return data;
    }catch(_){}
  }
  document.getElementById('status-pill').className='status-pill status-static';
  document.getElementById('status-text').textContent='DEMO · run pipeline first';
  return buildDemoData();
}

/* ── BOOT ───────────────────────────────────────────────── */
async function init(){
  UI_DATA=await loadData();
  const tickers=Object.keys(UI_DATA.tickers||{});
  if(!tickers.length){document.getElementById('loading').classList.add('hidden');return;}
  buildTickerBar(tickers);
  buildFilterBar(tickers);
  buildAllSignalsTab();
  await selectTicker(tickers[0]);
  document.getElementById('loading').classList.add('hidden');
}

/* ── FILTER BAR ─────────────────────────────────────────── */
function buildFilterBar(tickers){
  const counts={ALL:0,BUY:0,HOLD:0,SELL:0,WATCH:0,EXPIRED:0};
  tickers.forEach(t=>{
    counts.ALL++;
    const d=UI_DATA.tickers[t];const sig=d.signal||{};
    if(!sig.tier){counts.WATCH++;return;}
    const a=getTodayAction(d).action;
    if(a==='BUY')counts.BUY++;
    else if(a==='HOLD')counts.HOLD++;
    else if(a==='SELL')counts.SELL++;
    else if(a==='REVIEW')counts.EXPIRED++;
    else counts.WATCH++;
  });
  const bar=document.getElementById('filter-bar');if(!bar)return;
  const LABELS={ALL:'Tất cả',BUY:'▲ Mua vào',HOLD:'◆ Đang giữ',SELL:'▼ Bán ra',WATCH:'👀 Chờ',EXPIRED:'⏰ Hết hạn'};
  const COLS={ALL:'var(--muted)',BUY:'var(--green)',HOLD:'var(--blue)',SELL:'var(--red)',WATCH:'var(--amber)',EXPIRED:'var(--dim)'};
  bar.innerHTML=Object.entries(counts).map(([k,n])=>
    `<button class="filter-btn ${activeFilter===k?'active':''}" style="--fc:${COLS[k]}" onclick="applyFilter('${k}')">
       ${LABELS[k]} <span class="filter-count">${n}</span></button>`
  ).join('');
  filterTickerBar(tickers);
}

function applyFilter(f){
  activeFilter=f;
  buildFilterBar(Object.keys(UI_DATA.tickers||{}));
}

function filterTickerBar(tickers){
  tickers.forEach(t=>{
    const btn=document.getElementById(`tb-${t}`);if(!btn)return;
    if(activeFilter==='ALL'){btn.style.display='';return;}
    const d=UI_DATA.tickers[t];const a=getTodayAction(d).action;
    const map={BUY:'BUY',HOLD:'HOLD',SELL:'SELL',REVIEW:'EXPIRED',WATCH:'WATCH'};
    btn.style.display=(map[a]||'WATCH')===activeFilter?'':'none';
  });
}

/* ── TICKER BAR ─────────────────────────────────────────── */
function buildTickerBar(tickers){
  const bar=document.getElementById('ticker-bar');bar.innerHTML='';
  tickers.forEach(t=>{
    const d=UI_DATA.tickers[t];
    const sig=d.signal||{};const tier=sig.tier||0;
    const state=tier>0?getTodayAction(d):null;
    const dot=tier>0?`<span class="tier-dot tier-${tier}" style="background:${
      state?.action==='BUY'?'var(--green)':state?.action==='HOLD'?'var(--blue)':
      state?.action==='SELL'?'var(--red)':'var(--amber)'}"></span>`:'';
    const btn=document.createElement('button');
    btn.className='t-btn';btn.id=`tb-${t}`;
    btn.innerHTML=t+dot;btn.onclick=()=>selectTicker(t);
    bar.appendChild(btn);
  });
}

/* ── TICKER SWITCH WITH ANIMATION ───────────────────────── */
function scaleChart(raw,scale){
  return raw.map(r=>({...r,
    close:(r.close||0)*scale,open:(r.open||0)*scale,high:(r.high||0)*scale,low:(r.low||0)*scale,
    sma20:(r.sma20||0)*scale,sma50:(r.sma50||0)*scale,
    bb_upper:(r.bb_upper||0)*scale,bb_lower:(r.bb_lower||0)*scale,
    kalman:(r.kalman||0)*scale,upperBound:(r.upperBound||0)*scale,lowerBound:(r.lowerBound||0)*scale,
    sar:(r.sar||0)*scale,
  }));
}
function getOpts(){return{
  showSma:document.getElementById('toggle-sma')?.checked??true,
  showBB: document.getElementById('toggle-bb')?.checked??false,
  showSar:document.getElementById('toggle-sar')?.checked??false,
};}

async function selectTicker(ticker){
  if(ticker===currentTicker)return;
  const prev=[...chartData];
  if(currentTicker){const o=document.getElementById(`tb-${currentTicker}`);if(o)o.classList.remove('active');}
  currentTicker=ticker;
  const btn=document.getElementById(`tb-${ticker}`);if(btn)btn.classList.add('active');
  const d=UI_DATA.tickers[ticker];
  const scale=getPriceScale(d);
  const newData=scaleChart(d.chart||[],scale);
  renderStats(d);renderSignalCard(d,newData);renderFeatList(d);
  if(prev.length&&newData.length)Animator.start(prev,newData,getOpts());
  else{chartData=newData;redrawAll(chartData);}
  if(currentTab==='backtest')raf2(()=>renderBacktest(d));
}

function redrawChart(){if(Animator.running)Animator.stop();redrawAll(chartData);}
function redrawAll(data){drawPrice(data,getOpts());drawVol(data);drawRsi(data);drawXhair(hoverIdx,data);}

/* ── ANIMATION ENGINE ───────────────────────────────────── */
const Animator={
  _raf:null,_from:[],_to:[],_opts:{},_ts:null,DUR:380,
  get running(){return this._raf!==null;},
  start(from,to,opts){
    if(this._raf){cancelAnimationFrame(this._raf);this._raf=null;}
    const n=to.length;
    this._from=Array.from({length:n},(_,i)=>{const fi=from.length-n+i;return fi>=0?from[fi]:(from[0]||to[0]);});
    this._to=to;this._opts=opts;this._ts=null;
    const self=this;
    function tick(ts){
      if(!self._ts)self._ts=ts;
      const t=clamp((ts-self._ts)/self.DUR,0,1),e=easeOut(t);
      const ip=Array.from({length:n},(_,i)=>{
        const f=self._from[i],r=self._to[i];
        return{...r,
          close:lerp(f.close,r.close,e),sma20:lerp(f.sma20||0,r.sma20||0,e),sma50:lerp(f.sma50||0,r.sma50||0,e),
          bb_upper:lerp(f.bb_upper||0,r.bb_upper||0,e),bb_lower:lerp(f.bb_lower||0,r.bb_lower||0,e),
          upperBound:lerp(f.upperBound||0,r.upperBound||0,e),lowerBound:lerp(f.lowerBound||0,r.lowerBound||0,e),
          kalman:lerp(f.kalman||0,r.kalman||0,e),sar:lerp(f.sar||0,r.sar||0,e),
          volume:lerp(f.volume,r.volume,e),rsi:lerp(f.rsi||50,r.rsi||50,e),
          macd:lerp(f.macd||0,r.macd||0,e),relVolume:lerp(f.relVolume||1,r.relVolume||1,e),
        };
      });
      drawPrice(ip,self._opts);drawVol(ip);drawRsi(ip);
      if(t<1)self._raf=requestAnimationFrame(tick);
      else{chartData=self._to;self._raf=null;drawXhair(hoverIdx,chartData);}
    }
    this._raf=requestAnimationFrame(tick);
  },
  stop(){if(this._raf){cancelAnimationFrame(this._raf);this._raf=null;}chartData=this._to.length?this._to:chartData;}
};

/* ── CANVAS HELPERS ─────────────────────────────────────── */
const GEO={price:null,vol:null,rsi:null};
function setupCanvas(canvas,H){
  const dpr=window.devicePixelRatio||1;
  const W=canvas.parentElement?.offsetWidth||canvas.offsetWidth||600;
  canvas.width=W*dpr;canvas.height=H*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  return{ctx,W,H,dpr};
}
function syncOv(bId,oId){const b=document.getElementById(bId),o=document.getElementById(oId);if(b&&o){o.width=b.width;o.height=b.height;}}

/* ── PRICE CHART ────────────────────────────────────────── */
function drawPrice(data,opts={}){
  const canvas=document.getElementById('price-chart');if(!canvas||!data.length)return;
  const H=320;const{ctx,W,dpr}=setupCanvas(canvas,H);syncOv('price-chart','price-overlay');
  const pad={top:20,right:14,bottom:30,left:72};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  const allP=data.flatMap(r=>[r.close,r.upperBound||r.close,r.lowerBound||r.close,
    opts.showBB?(r.bb_upper||r.close):r.close,opts.showBB?(r.bb_lower||r.close):r.close]).filter(v=>v>0);
  let pMin=Math.min(...allP)*0.997,pMax=Math.max(...allP)*1.003;
  if(pMax<=pMin){pMin*=0.99;pMax*=1.01;}
  const xOf=i=>pad.left+(i/(data.length-1))*cW;
  const yOf=p=>pad.top+(1-(p-pMin)/(pMax-pMin))*cH;
  GEO.price={pad,xOf,yOf,cW,H,W,dpr};
  // Grid
  ctx.strokeStyle='#1a2340';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){
    const y=pad.top+(g/4)*cH;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();
    ctx.fillStyle='#5a6b8a';ctx.font='10px IBM Plex Mono,monospace';ctx.textAlign='right';
    ctx.fillText(fmt(pMax-(g/4)*(pMax-pMin)),pad.left-4,y+4);
  }
  const step=Math.max(1,Math.floor(data.length/6));
  ctx.fillStyle='#5a6b8a';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='center';
  data.forEach((r,i)=>{if(i%step===0)ctx.fillText(fmtD(r.date),xOf(i),H-pad.bottom+14);});
  // GARCH fill
  ctx.beginPath();
  data.forEach((r,i)=>{const y=yOf(r.upperBound||r.close);i===0?ctx.moveTo(xOf(i),y):ctx.lineTo(xOf(i),y);});
  [...data].reverse().forEach((r,i)=>ctx.lineTo(xOf(data.length-1-i),yOf(r.lowerBound||r.close)));
  ctx.closePath();ctx.fillStyle='rgba(77,159,255,0.05)';ctx.fill();
  ['upperBound','lowerBound'].forEach(k=>{
    ctx.beginPath();ctx.strokeStyle='rgba(74,85,104,0.4)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    let f=true;data.forEach((r,i)=>{if(!r[k])return;f?(ctx.moveTo(xOf(i),yOf(r[k])),f=false):ctx.lineTo(xOf(i),yOf(r[k]));});
    ctx.stroke();ctx.setLineDash([]);
  });
  if(opts.showBB){['bb_upper','bb_lower'].forEach(k=>{
    ctx.beginPath();ctx.strokeStyle='rgba(167,139,250,0.4)';ctx.lineWidth=1;ctx.setLineDash([3,4]);
    let f=true;data.forEach((r,i)=>{if(!r[k])return;f?(ctx.moveTo(xOf(i),yOf(r[k])),f=false):ctx.lineTo(xOf(i),yOf(r[k]));});
    ctx.stroke();ctx.setLineDash([]);
  });}
  if(opts.showSma){[['sma20','#ffb340',1.3],['sma50','rgba(34,211,238,.65)',1]].forEach(([k,col,lw])=>{
    ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=lw;let f=true;
    data.forEach((r,i)=>{if(!r[k])return;f?(ctx.moveTo(xOf(i),yOf(r[k])),f=false):ctx.lineTo(xOf(i),yOf(r[k]));});
    ctx.stroke();
  });}
  ctx.beginPath();ctx.strokeStyle='rgba(167,139,250,.7)';ctx.lineWidth=1.5;let fk=true;
  data.forEach((r,i)=>{if(!r.kalman)return;fk?(ctx.moveTo(xOf(i),yOf(r.kalman)),fk=false):ctx.lineTo(xOf(i),yOf(r.kalman));});ctx.stroke();
  ctx.beginPath();ctx.strokeStyle='#4d9fff';ctx.lineWidth=2;let fp=true;
  data.forEach((r,i)=>{fp?(ctx.moveTo(xOf(i),yOf(r.close)),fp=false):ctx.lineTo(xOf(i),yOf(r.close));});ctx.stroke();
  if(opts.showSar){data.forEach((r,i)=>{if(!r.sar)return;
    ctx.beginPath();ctx.arc(xOf(i),yOf(r.sar),2.5,0,Math.PI*2);
    ctx.fillStyle=r.sarBullish?'#00e5a0':'#ff4d6a';ctx.fill();
  });}
}

/* ── VOLUME CHART ───────────────────────────────────────── */
function drawVol(data){
  const canvas=document.getElementById('vol-chart');if(!canvas||!data.length)return;
  const H=100;const{ctx,W,dpr}=setupCanvas(canvas,H);syncOv('vol-chart','vol-overlay');
  const pad={top:6,right:14,bottom:20,left:72};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  const maxV=Math.max(...data.map(r=>r.volume||0))||1;
  const adv=data.reduce((s,r)=>s+(r.volume||0),0)/data.length;
  const bW=Math.max(1,cW/data.length-0.5);
  const xOf=i=>pad.left+(i/(data.length-1))*cW;
  GEO.vol={pad,xOf,cW,H,W,dpr};
  data.forEach((r,i)=>{
    const h=((r.volume||0)/maxV)*cH,y=pad.top+cH-h;
    ctx.fillStyle=(r.relVolume||1)>2?'rgba(0,229,160,.55)':'rgba(74,85,104,.45)';
    ctx.fillRect(xOf(i)-bW/2,y,bW,h);
  });
  const avgH=(adv/maxV)*cH;
  ctx.beginPath();ctx.strokeStyle='rgba(255,179,64,.4)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.moveTo(pad.left,pad.top+cH-avgH);ctx.lineTo(W-pad.right,pad.top+cH-avgH);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(255,179,64,.5)';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='right';
  ctx.fillText('ADV',pad.left-2,pad.top+cH-avgH+4);
}

/* ── RSI + MACD ─────────────────────────────────────────── */
function drawRsi(data){
  const canvas=document.getElementById('rsi-chart');if(!canvas||!data.length)return;
  const H=80;const{ctx,W,dpr}=setupCanvas(canvas,H);syncOv('rsi-chart','rsi-overlay');
  const pad={top:6,right:14,bottom:4,left:72};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  const xOf=i=>pad.left+(i/(data.length-1))*cW;
  const rY=v=>pad.top+(1-v/100)*cH;
  GEO.rsi={pad,xOf,cW,H,W,dpr};
  ctx.fillStyle='rgba(0,229,160,.05)';ctx.fillRect(pad.left,rY(30),cW,rY(0)-rY(30));
  ctx.fillStyle='rgba(255,77,106,.05)';ctx.fillRect(pad.left,rY(100),cW,rY(70)-rY(100));
  [30,50,70].forEach(v=>{
    ctx.beginPath();ctx.strokeStyle='#1a2340';ctx.lineWidth=1;ctx.moveTo(pad.left,rY(v));ctx.lineTo(W-pad.right,rY(v));ctx.stroke();
    ctx.fillStyle='#5a6b8a';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='right';ctx.fillText(v,pad.left-4,rY(v)+3);
  });
  const mMax=Math.max(...data.map(r=>Math.abs(r.macd||0)))||1;
  data.forEach((r,i)=>{
    const v=(r.macd||0)/mMax*0.4,bH=Math.abs(v)*cH,bY=v>=0?rY(50+Math.abs(v)*50):rY(50);
    ctx.fillStyle=v>=0?'rgba(0,229,160,.35)':'rgba(255,77,106,.35)';ctx.fillRect(xOf(i)-1.5,bY,3,bH);
  });
  ctx.beginPath();ctx.strokeStyle='#ffb340';ctx.lineWidth=1.5;let f=true;
  data.forEach((r,i)=>{if(r.rsi==null)return;f?(ctx.moveTo(xOf(i),rY(r.rsi)),f=false):ctx.lineTo(xOf(i),rY(r.rsi));});ctx.stroke();
  const last=data[data.length-1];
  if(last){const el=document.getElementById('rsi-value');if(el){
    el.textContent=`RSI ${(last.rsi||50).toFixed(1)}  MACD ${(last.macd||0)>=0?'+':''}${(last.macd||0).toFixed(0)}`;
    el.style.color=(last.rsi||50)<30?'var(--green)':(last.rsi||50)>70?'var(--red)':'var(--amber)';
  }}
}

/* ── SYNCHRONIZED CROSSHAIR ─────────────────────────────── */
function drawXhair(idx,data){
  hoverIdx=idx;
  _xh('price-overlay',GEO.price,idx,data,true);
  _xh('vol-overlay',  GEO.vol,  idx,data,false);
  _xh('rsi-overlay',  GEO.rsi,  idx,data,false);
}
function _xh(id,geo,idx,data,label){
  const canvas=document.getElementById(id);if(!canvas||!geo)return;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if(idx<0||!data||idx>=data.length)return;
  const{pad,xOf,H,W,dpr}=geo;ctx.save();ctx.scale(dpr,dpr);
  const x=xOf(idx);
  ctx.beginPath();ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(221,229,245,0.2)';ctx.lineWidth=1;
  ctx.moveTo(x,pad.top);ctx.lineTo(x,H-pad.bottom);ctx.stroke();ctx.setLineDash([]);
  if(label&&geo.yOf){
    const r=data[idx],y=geo.yOf(r.close);
    ctx.beginPath();ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(221,229,245,0.12)';
    ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);
    ctx.fillStyle='#4d9fff';ctx.strokeStyle='#060810';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
    const lbl=fmt(r.close)+' ₫';ctx.font='bold 10px IBM Plex Mono,monospace';
    const tw=ctx.measureText(lbl).width+8;ctx.fillStyle='#4d9fff';ctx.fillRect(pad.left-tw-2,y-9,tw,18);
    ctx.fillStyle='#060810';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(lbl,pad.left-5,y);
  }
  ctx.restore();
}

/* ── MOUSE EVENTS ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded',()=>{
  ['price-chart-wrap','vol-chart-wrap','rsi-chart-wrap'].forEach(wrapId=>{
    const wrap=document.getElementById(wrapId);if(!wrap)return;
    wrap.addEventListener('mousemove',e=>{
      if(!chartData.length||!GEO.price)return;
      const rect=document.getElementById('price-chart').getBoundingClientRect();
      const idx=clamp(Math.round((e.clientX-rect.left-GEO.price.pad.left)/GEO.price.cW*(chartData.length-1)),0,chartData.length-1);
      drawXhair(idx,chartData);
      const r=chartData[idx],prev=chartData[idx-1];
      const chg=prev?(r.close-prev.close)/prev.close*100:0;
      document.getElementById('tt-date').textContent=r.date;
      document.getElementById('tt-close').textContent=fmt(r.close)+' ₫';
      document.getElementById('tt-chg').textContent=fmtP(chg);
      document.getElementById('tt-chg').style.color=chg>=0?'var(--green)':'var(--red)';
      document.getElementById('tt-vol').textContent=(r.volume/1e6).toFixed(2)+'M';
      document.getElementById('tt-rsi').textContent=(r.rsi||0).toFixed(1);
      document.getElementById('tt-adx').textContent=(r.adx||0).toFixed(1);
      const vh=document.getElementById('vol-hover');
      if(vh)vh.textContent=`KL: ${(r.volume/1e6).toFixed(2)}M  ×${(r.relVolume||1).toFixed(2)}`;
      const tt=document.getElementById('chart-tooltip');
      tt.style.display='block';tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
    });
    wrap.addEventListener('mouseleave',()=>{
      drawXhair(-1,chartData);
      document.getElementById('chart-tooltip').style.display='none';
      const vh=document.getElementById('vol-hover');if(vh)vh.textContent='';
    });
  });
});

/* ── RENDER: STATS ──────────────────────────────────────── */
function renderStats(d){
  const ind=d.indicators||{},scale=getPriceScale(d);
  const price=sp(d.price,scale),chgP=d.changePercent||0;
  document.getElementById('stats-grid').innerHTML=`
    <div class="stat-card" style="--accent-color:${chgP>=0?'var(--green)':'var(--red)'}">
      <div class="stat-label">${d.ticker} · ${d.sector}</div>
      <div class="stat-value">${fmt(price)}<span style="font-size:13px;color:var(--muted)"> ₫</span></div>
      <div class="stat-sub ${chgP>=0?'up':'down'}">${fmtP(chgP)} hôm nay</div>
    </div>
    <div class="stat-card" style="--accent-color:${ind.hurst>0.55?'var(--green)':ind.hurst<0.45?'var(--red)':'var(--amber)'}">
      <div class="stat-label">Market Regime (Hurst)</div>
      <div class="stat-value ${ind.hurst>0.55?'up':ind.hurst<0.45?'down':'amb'}">${(ind.hurst||0.5).toFixed(3)}</div>
      <div class="stat-sub neu">${ind.regime||'—'}</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--blue)">
      <div class="stat-label">ADX (Trend Strength)</div>
      <div class="stat-value ${(ind.adx14||0)>25?'up':'neu'}">${(ind.adx14||0).toFixed(1)}</div>
      <div class="stat-sub neu">${(ind.adx14||0)>40?'Rất mạnh':(ind.adx14||0)>25?'Mạnh':'Yếu'}</div>
    </div>
    <div class="stat-card" style="--accent-color:var(--purple)">
      <div class="stat-label">Wyckoff Phase</div>
      <div class="stat-value" style="color:var(--purple)">${ind.wyckoffPhase||'B'}</div>
      <div class="stat-sub neu">Uptrend Quality: ${Math.round((ind.uptrend_quality||0)*100)}%</div>
    </div>`;
}

/* ── RENDER: SIGNAL CARD ────────────────────────────────── */
function renderSignalCard(d,scaledChart){
  const sig=d.signal||{},wrap=document.getElementById('signal-card-wrap');
  const scale=getPriceScale(d),currentPrice=sp(d.price,scale);

  if(!sig.tier||sig.signal==='NONE'){
    wrap.innerHTML=`<div class="card" style="padding:20px;text-align:center;">
      <div style="color:var(--muted);font-family:monospace;font-size:12px;padding:16px 0;">
        Không có tín hiệu · Chờ cơ hội<br><span style="font-size:10px;opacity:.5;">Run pipeline to generate signals</span>
      </div></div>`;return;
  }

  const state=getSignalState(sig,currentPrice,scale);
  const basePriceRaw=parseFloat(sig.base_price||sig.current_price||0);
  const basePrice=sp(basePriceRaw,scale)||currentPrice;
  const{entryLo,entryHi,target,stopLoss,rewardPct,riskPct,rrRatio}=computeSignalPrices(sig,basePrice);
  const rrBar=clamp((rrRatio-0.5)/2.5*100,5,95);

  const stateAction=state&&state.action;
  const dispSig=stateAction==='SELL'?'SELL':stateAction==='HOLD'?'HOLD':
                stateAction==='REVIEW'?'NONE':stateAction==='WATCH'?'NONE':sig.signal||'NONE';
  const badgeM={BUY:'badge-buy',SELL:'badge-sell',HOLD:'badge-hold',NONE:'badge-none'};
  const labelM={BUY:'▲ MUA',SELL:'▼ BÁN',HOLD:'◆ GIỮ',NONE:'— XEM LẠI'};

  let dots='';for(let i=0;i<9;i++)dots+=`<div class="sig-dot ${i<(sig.n_signals||0)?'on':''}"></div>`;

  const heldStr=state&&state.held_pct!=null?
    `P&L từ giá vào: <strong style="color:${parseFloat(state.held_pct)>=0?'var(--green)':'var(--red)'}">
    ${parseFloat(state.held_pct)>=0?'+':''}${state.held_pct}%</strong>` : '';

  wrap.innerHTML=`
    <div class="signal-card" style="--sig-color:${dispSig==='SELL'?'var(--red)':dispSig==='HOLD'?'var(--blue)':'var(--green)'};">
      <div class="sig-header">
        <div>
          <div class="sig-ticker">${d.ticker}</div><div class="sig-name">${d.name}</div>
          <div class="sig-date">Tín hiệu: ${sig.date||'—'} · Hết: ${sig.exit_date||'—'}</div>
          ${state?`<span class="state-badge ${state.cls}">${state.label}</span>`:''}
        </div>
        <div style="text-align:right;">
          <span class="sig-badge ${badgeM[dispSig]||'badge-none'}">${labelM[dispSig]||'—'}</span>
          <div style="font-size:10px;color:var(--muted);margin-top:6px;">${sig.tier_label||`TIER ${sig.tier}`}</div>
        </div>
      </div>
      <div class="sig-body">
        <div style="margin-bottom:10px;">
          <span class="strat-tag">${sig.strategy_name||sig.primary_strategy||'—'}</span>
          <span class="hold-tag">⏱ T+${sig.hold_days||'?'}</span>
        </div>
        <div class="price-2col">
          <div class="price-box"><div class="lb">Giá hiện tại</div><div class="vl p-now">${fmt(currentPrice)} ₫</div></div>
          <div class="price-box"><div class="lb">Giá vào lệnh (base)</div><div class="vl" style="color:var(--amber)">${fmt(basePrice)} ₫</div></div>
          <div class="price-box"><div class="lb">Mục tiêu (+${rewardPct.toFixed(1)}%)</div><div class="vl p-target">${fmt(target)} ₫</div></div>
          <div class="price-box"><div class="lb">Cắt lỗ (−${riskPct.toFixed(1)}%)</div><div class="vl p-stop">${fmt(stopLoss)} ₫</div></div>
        </div>
        <div class="rr-wrap">
          <div class="rr-labels"><span>Rủi ro −${riskPct.toFixed(1)}%</span><span style="color:var(--text)">R:R = ${rrRatio.toFixed(2)}:1</span><span>LN +${rewardPct.toFixed(1)}%</span></div>
          <div class="rr-track"><div class="rr-fill" style="width:${rrBar}%"></div></div>
        </div>
        <div class="ind-3col">
          <div class="ind-box"><div class="ib-l">RSI 14</div><div class="ib-v ${(sig.rsi_14||50)<30?'up':(sig.rsi_14||50)>70?'down':'neu'}">${(sig.rsi_14||0).toFixed(1)}</div></div>
          <div class="ind-box"><div class="ib-l">ADX 14</div><div class="ib-v ${(sig.adx_14||0)>25?'up':'neu'}">${(sig.adx_14||0).toFixed(1)}</div></div>
          <div class="ind-box"><div class="ib-l">KL/TB</div><div class="ib-v ${(sig.rel_volume||1)>2?'up':'neu'}">${(sig.rel_volume||0).toFixed(2)}×</div></div>
        </div>
        ${sig.quality_score?`
        <div class="quality-bar-wrap">
          <div class="quality-label">
            <span>Điểm chất lượng</span>
            <span style="color:${(sig.quality_score||0)>=70?'var(--green)':(sig.quality_score||0)>=45?'var(--amber)':'var(--red)'}">${sig.quality_score}/100</span>
          </div>
          <div class="quality-bar-track"><div class="quality-bar-fill" style="width:${sig.quality_score}%"></div></div>
        </div>`:''}
        <div class="conf-row">
          <div class="conf-cell"><div class="conf-label">Xác suất</div><div class="conf-val">${_dispProb(sig)}%</div></div>
          <div class="conf-cell"><div class="conf-label">Tín hiệu</div><div class="conf-val">${sig.n_signals||0}/9</div><div class="dots-row">${dots}</div></div>
          <div class="conf-cell"><div class="conf-label">Win rate</div><div class="conf-val">${(sig.win_rate_hist||0).toFixed(1)}%</div></div>
        </div>
        <div class="cancel-box">
          <strong>🛑 Điều kiện hủy tín hiệu</strong>
          Thoát ngay nếu đóng cửa dưới <strong>${fmt(stopLoss)} ₫</strong> · ${heldStr}
        </div>
      </div>
    </div>`;
}

/* ── RENDER: FEATURES ───────────────────────────────────── */
function renderFeatList(d){
  const ind=d.indicators||{};
  const row=(lbl,val,cls='')=>`<div class="feat-row"><span class="feat-label">${lbl}</span><span class="feat-value ${cls}">${val}</span></div>`;
  document.getElementById('feat-list').innerHTML=
    row('GARCH σ (ann.)',`${(ind.garchSigma||0).toFixed(1)}%`,'neu')+
    row('VSA Ratio',`${(ind.vsaRatio||0).toFixed(3)}`,'neu')+
    row('VWAP Deviation',`${(ind.vwapDeviation||0)>=0?'+':''}${(ind.vwapDeviation||0).toFixed(2)}%`,(ind.vwapDeviation||0)>0?'up':'down')+
    row('Ceiling Demand',`${ind.ceilingDemand||0}`,'neu')+
    row('Bull Trap',ind.bullTrapFlag?'DETECTED':'CLEAR',ind.bullTrapFlag?'down':'up')+
    row('Bear Trap',ind.bearTrapFlag?'DETECTED':'CLEAR',ind.bearTrapFlag?'up':'down')+
    row('FOMO Signal',ind.fomoSignal?'ACTIVE':'CLEAR',ind.fomoSignal?'down':'up')+
    row('T+2.5 Risk',`${(ind.t25Risk||0).toFixed(1)}%`,(ind.t25Risk||0)>6?'down':'up');
  document.getElementById('macro-list').innerHTML=
    row('Breadth > 50-SMA',`${(ind.breadthPctAbove50||50).toFixed(1)}%`,(ind.breadthPctAbove50||50)>50?'up':'down')+
    row('A/D Ratio 10d',`${(ind.adRatio10d||1).toFixed(2)}`,(ind.adRatio10d||1)>1?'up':'down')+
    row('Interbank Rate',`${(ind.interbankRate||3.5).toFixed(1)}%`,(ind.interbankRate||3.5)<4?'up':'down')+
    row('USD/VND 10d',`${(ind.usdVndTrend||0)>=0?'+':''}${(ind.usdVndTrend||0).toFixed(2)}%`,(ind.usdVndTrend||0)<0?'up':'down')+
    row('VN-Index MA',`${(ind.vnIndexScore||0.5).toFixed(2)}`,(ind.vnIndexScore||0.5)>0.5?'up':'down');
}

/* ── RENDER: BACKTEST ───────────────────────────────────── */
function renderBacktest(d){
  const m=d.metrics||{},bt=d.backtest||{};
  const equity=bt.equity||[],trades=bt.trades||[];
  const scale=getPriceScale(d),c=v=>v>=0?'up':'down';
  document.getElementById('bt-stats-grid').innerHTML=`
    <div class="stat-card" style="--accent-color:${(m.ann_return||0)>=0?'var(--green)':'var(--red)'}">
      <div class="stat-label">Annual Return (CAGR)</div><div class="stat-value ${c(m.ann_return||0)}">${((m.ann_return||0)*100).toFixed(1)}%</div></div>
    <div class="stat-card" style="--accent-color:var(--blue)">
      <div class="stat-label">Sharpe Ratio</div><div class="stat-value ${(m.sharpe||0)>=1?'up':'down'}">${(m.sharpe||0).toFixed(2)}</div></div>
    <div class="stat-card" style="--accent-color:var(--amber)">
      <div class="stat-label">Win Rate</div><div class="stat-value">${((m.win_rate||0)*100).toFixed(1)}%</div>
      <div class="stat-sub neu">${m.n_trades||0} giao dịch</div></div>
    <div class="stat-card" style="--accent-color:var(--purple)">
      <div class="stat-label">Profit Factor</div><div class="stat-value ${(m.profit_factor||0)>=1.5?'up':'down'}">${(m.profit_factor||0).toFixed(2)}</div>
      <div class="stat-sub ${c(m.max_drawdown||0)}">DD: ${((m.max_drawdown||0)*100).toFixed(1)}%</div></div>`;

  const posEl=document.getElementById('bt-position-badge');
  const sig=d.signal||{};
  if(sig.tier>0&&posEl){
    const cp=sp(d.price,scale);const basePriceRaw=parseFloat(sig.base_price||0);
    const base=sp(basePriceRaw,scale)||cp;const{target,stopLoss}=computeSignalPrices(sig,base);
    const state=getSignalState(sig,cp,scale);
    const heldPct=base>0?((cp-base)/base*100).toFixed(1):'0';
    posEl.innerHTML=`<span style="color:${state?.action==='SELL'?'var(--red)':state?.action==='HOLD'?'var(--blue)':'var(--green)'}">
      ${state?.label||'—'}</span>&nbsp;Base ${fmt(base)} ₫&nbsp;→&nbsp;Target ${fmt(target)} ₫&nbsp;Stop ${fmt(stopLoss)} ₫
      <span style="color:${parseFloat(heldPct)>=0?'var(--green)':'var(--red)'}">&nbsp;${parseFloat(heldPct)>=0?'+':''}${heldPct}%</span>`;
  }else if(posEl)posEl.textContent='';

  const cvs=document.getElementById('bt-equity-chart');if(!cvs)return;
  if(!equity.length){
    const pw=cvs.parentElement?.offsetWidth||600;cvs.width=pw;cvs.height=280;cvs.style.width='100%';
    const ctx=cvs.getContext('2d');ctx.fillStyle='#5a6b8a';ctx.font='12px IBM Plex Mono,monospace';ctx.textAlign='center';
    ctx.fillText('Chạy 06_export.py để có dữ liệu backtest',pw/2,140);
    attachBacktestHover(equity);return;
  }
  const dpr=window.devicePixelRatio||1;
  const W=cvs.parentElement?.offsetWidth||cvs.offsetWidth||800,H=280;
  cvs.width=W*dpr;cvs.height=H*dpr;cvs.style.width='100%';cvs.style.height=H+'px';
  const ctx=cvs.getContext('2d');ctx.scale(dpr,dpr);
  const pad={top:20,right:16,bottom:30,left:80};
  const cW=W-pad.left-pad.right,cH=H-pad.top-pad.bottom;
  const vals=equity.map(e=>e.equity);
  const eMin=Math.min(...vals),eMax=Math.max(...vals);
  const xOf=i=>pad.left+(i/(equity.length-1))*cW;
  const yOf=v=>pad.top+(1-(v-eMin)/(eMax-eMin||1))*cH;
  for(let g=0;g<=4;g++){
    const y=pad.top+(g/4)*cH;ctx.strokeStyle='#1a2340';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();
    ctx.fillStyle='#5a6b8a';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='right';
    ctx.fillText(((eMax-(g/4)*(eMax-eMin))/1e6).toFixed(1)+'M',pad.left-4,y+3);
  }
  ctx.beginPath();equity.forEach((e,i)=>i===0?ctx.moveTo(xOf(i),yOf(e.equity)):ctx.lineTo(xOf(i),yOf(e.equity)));
  ctx.lineTo(xOf(equity.length-1),pad.top+cH);ctx.lineTo(xOf(0),pad.top+cH);ctx.closePath();
  ctx.fillStyle='rgba(139,92,246,.1)';ctx.fill();
  ctx.beginPath();ctx.strokeStyle='#8b5cf6';ctx.lineWidth=2;
  equity.forEach((e,i)=>i===0?ctx.moveTo(xOf(i),yOf(e.equity)):ctx.lineTo(xOf(i),yOf(e.equity)));ctx.stroke();
  const startY=yOf(100_000_000);
  ctx.beginPath();ctx.strokeStyle='rgba(255,179,64,.3)';ctx.lineWidth=1;ctx.setLineDash([5,5]);
  ctx.moveTo(pad.left,startY);ctx.lineTo(W-pad.right,startY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(255,179,64,.5)';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='left';
  ctx.fillText('100M',pad.left+4,startY-3);
  trades.slice(-60).forEach(t=>{
    const d2=new Date(t.entry_date||'');const ei=equity.findIndex(e=>new Date(e.date)>=d2);if(ei<0)return;
    const x=xOf(clamp(ei,0,equity.length-1)),y2=yOf(equity[ei]?.equity||eMin);
    ctx.beginPath();ctx.arc(x,y2,3,0,Math.PI*2);
    ctx.fillStyle=(t.pnl_pct||0)>=0?'rgba(0,229,160,.7)':'rgba(255,77,106,.7)';ctx.fill();
  });
  attachBacktestHover(equity);

  const tbody=document.getElementById('trade-tbody');
  if(!trades.length){tbody.innerHTML='<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px;">Chưa có giao dịch</td></tr>';return;}
  tbody.innerHTML=trades.slice(-40).map(t=>{
    const pnl=(t.pnl_pct||0)*100,row=pnl>=0?'win-row':'loss-row';
    return`<tr class="${row}"><td>${(t.entry_date||'').slice(0,10)}</td><td>${(t.exit_date||'').slice(0,10)}</td>
      <td>${fmt((t.entry_price||0)*scale)}</td><td>${fmt((t.exit_price||0)*scale)}</td>
      <td>${(pnl>=0?'+':'')+pnl.toFixed(2)}%</td>
      <td style="color:var(--purple);font-size:10px;">${t.strategy||'—'}</td>
      <td style="color:var(--amber)">T${t.tier||'?'}</td>
      <td style="color:var(--muted);font-size:10px;">${t.reason||'—'}</td></tr>`;
  }).join('');
}

/* ── BACKTEST HOVER ─────────────────────────────────────── */
function attachBacktestHover(equity){
  const cvs=document.getElementById('bt-equity-chart');if(!cvs||cvs._hov)return;
  cvs._hov=true;
  let tt=document.getElementById('bt-tt');
  if(!tt){tt=document.createElement('div');tt.id='bt-tt';
    tt.style.cssText='position:fixed;pointer-events:none;background:var(--surface2);border:1px solid var(--border2);'+
      'border-radius:6px;padding:8px 12px;font-family:IBM Plex Mono,monospace;font-size:11px;display:none;z-index:600;'+
      'box-shadow:0 4px 20px rgba(0,0,0,.5);min-width:150px;';
    document.body.appendChild(tt);}
  cvs.addEventListener('mousemove',e=>{
    if(!equity||!equity.length)return;
    const rect=cvs.getBoundingClientRect();
    const pad={top:20,right:16,bottom:30,left:80};
    const cW=rect.width-pad.left-pad.right;
    const idx=clamp(Math.round((e.clientX-rect.left-pad.left)/cW*(equity.length-1)),0,equity.length-1);
    const pt=equity[idx],prev=equity[idx-1];
    const ret=prev?(pt.equity-prev.equity)/prev.equity*100:0;
    const total=(pt.equity/100_000_000-1)*100;
    tt.innerHTML=`<div style="color:var(--muted);font-size:10px;margin-bottom:5px;">${pt.date}</div>
      <div style="display:flex;justify-content:space-between;gap:14px;"><span style="color:var(--muted)">Vốn</span><span>${(pt.equity/1e6).toFixed(2)}M ₫</span></div>
      <div style="display:flex;justify-content:space-between;gap:14px;margin-top:2px;"><span style="color:var(--muted)">Phiên</span><span style="color:${ret>=0?'var(--green)':'var(--red)'}">${ret>=0?'+':''}${ret.toFixed(2)}%</span></div>
      <div style="display:flex;justify-content:space-between;gap:14px;margin-top:2px;"><span style="color:var(--muted)">Tổng</span><span style="color:${total>=0?'var(--green)':'var(--red)'}">${total>=0?'+':''}${total.toFixed(1)}%</span></div>`;
    tt.style.display='block';tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
  });
  cvs.addEventListener('mouseleave',()=>{tt.style.display='none';});
}

/* ── ALL SIGNALS TAB ────────────────────────────────────── */
function buildAllSignalsTab(){
  const wrap=document.getElementById('all-signals-wrap');
  const summary=(UI_DATA.daily_summary||[]).slice(0,24);
  if(!summary.length){wrap.innerHTML='<p style="color:var(--muted);padding:20px;font-family:monospace;">No active signals. Run pipeline.</p>';return;}
  wrap.innerHTML='';
  summary.forEach(s=>{
    const d=UI_DATA.tickers[s.ticker];if(!d)return;
    const sig=d.signal||{},scale=getPriceScale(d),cp=sp(d.price,scale);
    const basePriceRaw=parseFloat(sig.base_price||cp);const base=sp(basePriceRaw,scale)||cp;
    const tier=s.tier||sig.tier||0;const col=tier===1?'var(--green)':tier===2?'var(--amber)':'var(--blue)';
    const state=getSignalState(sig,cp,scale);
    const{target,rewardPct}=computeSignalPrices(sig,base);
    const card=document.createElement('div');card.className='signal-card';
    card.style.setProperty('--sig-color',state?.action==='SELL'?'var(--red)':state?.action==='HOLD'?'var(--blue)':col);
    card.style.cursor='pointer';
    card.innerHTML=`
      <div class="sig-header">
        <div><div class="sig-ticker">${s.ticker}</div><div class="sig-name">${d.name}</div></div>
        ${state?`<span class="state-badge ${state.cls}" style="margin-top:4px;">${state.label}</span>`:
                `<span class="sig-badge badge-buy">▲ MUA</span>`}
      </div>
      <div class="sig-body" style="padding:10px 14px;">
        <span class="strat-tag">${sig.strategy_name||s.strat||'—'}</span>
        <span class="hold-tag">T+${sig.hold_days||'?'}</span>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:10px;">
          <div><div style="font-size:9px;color:var(--muted);">PRICE</div><div style="font-family:monospace;">${fmt(cp)} ₫</div></div>
          <div><div style="font-size:9px;color:var(--muted);">TARGET</div><div style="font-family:monospace;color:var(--green)">+${rewardPct.toFixed(1)}%</div></div>
          <div><div style="font-size:9px;color:var(--muted);">PROB</div><div style="font-family:monospace;color:${col}">${_dispProb(sig)}%</div></div>
        </div>
      </div>`;
    card.onclick=()=>{switchTab('live');selectTicker(s.ticker);};
    wrap.appendChild(card);
  });
}

/* ── TAB SWITCHING ──────────────────────────────────────── */
function switchTab(name){
  currentTab=name;
  ['live','backtest','signals'].forEach(t=>{document.getElementById(`tab-${t}`).style.display=t===name?'block':'none';});
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{b.classList.toggle('active',['live','backtest','signals'][i]===name);});
  if(name==='backtest'&&currentTicker&&UI_DATA)raf2(()=>renderBacktest(UI_DATA.tickers[currentTicker]));
}

window.addEventListener('resize',()=>{
  if(!currentTicker||!UI_DATA)return;
  redrawAll(chartData);
  if(currentTab==='backtest')raf2(()=>renderBacktest(UI_DATA.tickers[currentTicker]));
});

/* ── DEMO DATA ──────────────────────────────────────────── */
function buildDemoData(){
  const TICKERS=['HPG','VCB','VHM','VNM','TCB','FPT','MWG','SSI','GAS','MSN'];
  const META={HPG:{name:'Tập đoàn Hòa Phát',sector:'Steel',price:26400},VCB:{name:'Vietcombank',sector:'Banking',price:85000},
    VHM:{name:'Vinhomes',sector:'Real Estate',price:42000},VNM:{name:'Vinamilk',sector:'Consumer',price:68000},
    TCB:{name:'Techcombank',sector:'Banking',price:21000},FPT:{name:'FPT Corporation',sector:'Technology',price:95000},
    MWG:{name:'Mobile World',sector:'Retail',price:48000},SSI:{name:'SSI Securities',sector:'Securities',price:32000},
    GAS:{name:'PetroVN Gas',sector:'Energy',price:78000},MSN:{name:'Masan Group',sector:'Consumer',price:71000}};
  const STRATS=[{key:'RSI_OVERSOLD',name:'RSI Quá Bán',hold:60,wr:74.3,tgt:0.15,sl:0.05},
    {key:'PULLBACK_BUY',name:'Mua Vào Nhịp Điều Chỉnh',hold:20,wr:71.0,tgt:0.10,sl:0.05},
    {key:'DMI_WAVE',name:'Lướt Sóng DMI',hold:10,wr:70.0,tgt:0.08,sl:0.04},
    {key:'UPTREND',name:'Uptrend Chắc Chắn',hold:60,wr:68.0,tgt:0.15,sl:0.06}];
  function mkS(baseP,n=120){const rows=[];let p=baseP;const now=new Date();
    for(let i=n;i>=0;i--){const d=new Date(now);d.setDate(d.getDate()-i);
      if(d.getDay()===0||d.getDay()===6)continue;
      p=Math.max(p*(1+(Math.random()-0.48)*0.015),baseP*0.5);
      const vol=Math.floor(Math.exp(14.5+Math.random()*0.8));
      rows.push({date:d.toISOString().slice(0,10),open:Math.round(p*(1+Math.random()*0.005)),
        high:Math.round(p*1.008),low:Math.round(p*0.992),close:Math.round(p),volume:vol,
        sma20:Math.round(p*0.996),sma50:Math.round(p*0.988),bb_upper:Math.round(p*1.02),bb_lower:Math.round(p*0.98),
        kalman:Math.round(p*0.999),upperBound:Math.round(p*1.03),lowerBound:Math.round(p*0.97),
        rsi:30+Math.random()*40,adx:15+Math.random()*35,macd:(Math.random()-0.5)*0.003*p,
        relVolume:0.5+Math.random()*3,sar:Math.round(p*0.97),sarBullish:Math.random()>0.4?1:0,hurst:0.4+Math.random()*0.2});
    }return rows;}
  const out={generated_at:new Date().toISOString(),tickers:{},daily_summary:[]};
  TICKERS.forEach((t,idx)=>{
    const m=META[t],chart=mkS(m.price);const last=chart[chart.length-1],prev=chart[chart.length-2]||last;
    const strat=STRATS[idx%STRATS.length];const tier=1+(idx%3);const nSig=tier===1?3:tier===2?2:1;
    // Vary the base price to exercise different signal states
    const daysBack=[3,12,25,50,90,130][idx%6];
    const pastIdx=Math.max(0,chart.length-1-daysBack);
    const basePrice=chart[pastIdx].close;
    const target=roundTick(basePrice*(1+strat.tgt));const stopLoss=roundTick(basePrice*(1-strat.sl));
    const bday=pd_bdate(chart[pastIdx].date,strat.hold);
    const qual=35+Math.random()*55;
    let cap=100_000_000;const eq=[];
    chart.forEach((r,i)=>{if(i%3===0){cap*=1+(Math.random()-0.45)*0.025;eq.push({date:r.date,equity:Math.round(cap)});}});
    const trades=Array.from({length:8},()=>{const ei=Math.floor(Math.random()*(chart.length-15));const ep=chart[ei].close;
      const xp=ep*(1+(Math.random()>0.65?1:-1)*Math.random()*0.08);
      return{entry_date:chart[ei].date,exit_date:chart[Math.min(ei+7,chart.length-1)].date,entry_price:ep,exit_price:Math.round(xp),
        pnl_pct:(xp-ep)/ep,strategy:strat.key,tier,reason:Math.random()>0.7?'stop_loss':'hold_period'};});
    out.tickers[t]={ticker:t,name:m.name,sector:m.sector,price:last.close,change:last.close-prev.close,
      changePercent:(last.close-prev.close)/prev.close*100,chart,
      signal:{ticker:t,signal:'BUY',tier,strategy_name:strat.name,primary_strategy:strat.key,
        n_signals:nSig,hold_days:strat.hold,date:chart[pastIdx].date,exit_date:bday,
        win_rate_hist:strat.wr,dir_prob:strat.wr+(nSig-1)*5,model_dir_prob:strat.wr+(nSig-1)*5,
        rsi_14:last.rsi,adx_14:last.adx,rel_volume:last.relVolume,quality_score:Math.round(qual),
        base_price:basePrice,current_price:last.close,target,stop_loss:stopLoss,
        entry_lo:roundTick(basePrice*0.98),entry_hi:roundTick(basePrice*1.01),
        tier_label:tier===1?'TIER 1 — ĐỘ TIN CẬY CAO NHẤT':tier===2?'TIER 2 — ĐỘ TIN CẬY CAO':'TIER 3 — CHỜ XÁC NHẬN'},
      indicators:{hurst:0.4+Math.random()*0.3,regime:'Trending',adx14:last.adx,rsi14:last.rsi,
        garchSigma:15+Math.random()*10,vsaRatio:Math.random()*0.4,vwapDeviation:(Math.random()-0.5)*3,
        uptrend_quality:0.4+Math.random()*0.6,wyckoffPhase:['B','C','D','E'][idx%4],
        breadthPctAbove50:40+Math.random()*30,adRatio10d:0.8+Math.random()*0.8,interbankRate:3+Math.random()*2,
        usdVndTrend:(Math.random()-0.5)*2,vnIndexScore:0.3+Math.random()*0.7,
        bullTrapFlag:false,bearTrapFlag:false,fomoSignal:false,ceilingDemand:0,t25Risk:2+Math.random()*6},
      metrics:{n_trades:trades.length,win_rate:0.5+Math.random()*0.3,ann_return:-0.05+Math.random()*0.4,
        max_drawdown:-(0.05+Math.random()*0.2),sharpe:0.5+Math.random()*2,profit_factor:1.2+Math.random()*1.5},
      backtest:{equity:eq,trades}};
    if(tier<=2)out.daily_summary.push({ticker:t,tier,strat:strat.key,date:chart[pastIdx].date,
      prob:strat.wr+(nSig-1)*5,base_price:basePrice,target,stop_loss:stopLoss,exit_date:bday});
  });
  return out;
}
function pd_bdate(start,n){const d=new Date(start);let c=0;while(c<n){d.setDate(d.getDate()+1);if(d.getDay()!==0&&d.getDay()!==6)c++;}return d.toISOString().slice(0,10);}

init();