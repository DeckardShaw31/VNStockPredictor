import { addDays, format, subDays } from 'date-fns';

export interface StockData {
  date: string;
  price: number;
  kalman: number;
  upperBound: number;
  lowerBound: number;
  volume: number;
  foreignFlow: number;
}

export interface StockInfo {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
}

export const VN100_STOCKS: StockInfo[] = [
  { symbol: 'HPG', name: 'Hoa Phat Group', sector: 'Steel', price: 30500, change: 500, changePercent: 1.67 },
  { symbol: 'VCB', name: 'Vietcombank', sector: 'Banking', price: 95000, change: -1000, changePercent: -1.04 },
  { symbol: 'SSI', name: 'SSI Securities', sector: 'Securities', price: 38200, change: 1200, changePercent: 3.24 },
  { symbol: 'FPT', name: 'FPT Corporation', sector: 'Technology', price: 115000, change: 2500, changePercent: 2.22 },
  { symbol: 'MWG', name: 'Mobile World', sector: 'Retail', price: 52000, change: -500, changePercent: -0.95 },
];

export function generateMockData(symbol: string, days: number = 60): StockData[] {
  const data: StockData[] = [];
  const today = new Date();
  
  let basePrice = VN100_STOCKS.find(s => s.symbol === symbol)?.price || 50000;
  let currentPrice = basePrice * 0.8; // Start lower
  let kalman = currentPrice;
  
  for (let i = days; i >= 0; i--) {
    const date = subDays(today, i);
    
    // Random walk with drift
    const drift = (Math.random() - 0.45) * (basePrice * 0.02);
    currentPrice = currentPrice + drift;
    
    // Kalman filter smoothing (lagged)
    kalman = kalman * 0.8 + currentPrice * 0.2;
    
    // GARCH volatility simulation (expanding/contracting bounds)
    const volatilityBase = basePrice * 0.03;
    const garchVol = volatilityBase * (1 + Math.sin(i / 5) * 0.5 + Math.random() * 0.2);
    
    const upperBound = kalman + garchVol * 1.96;
    const lowerBound = kalman - garchVol * 1.96;
    
    const volume = Math.floor(Math.random() * 10000000) + 2000000;
    const foreignFlow = (Math.random() - 0.4) * volume * 0.2; // Net foreign
    
    data.push({
      date: format(date, 'MMM dd'),
      price: Math.round(currentPrice),
      kalman: Math.round(kalman),
      upperBound: Math.round(upperBound),
      lowerBound: Math.round(lowerBound),
      volume,
      foreignFlow: Math.round(foreignFlow)
    });
  }
  
  return data;
}

export interface ModelSignals {
  hurst: number;
  regime: 'Trending' | 'Mean-Reverting' | 'Chaotic';
  vnIndexScore: number;
  sectorRS: number;
  vsaRatio: number;
  vwapDeviation: number;
  ceilingDemand: number;
  garchSigma: number;
  directionProb: number;
  t25Risk: number;
  action: 'BUY' | 'SELL' | 'HOLD';
  actionReason: string;
  usdVndTrend: number;
  interbankRate: number;
  bullTrapFlag: boolean;
  bearTrapFlag: boolean;
  fomoSignal: boolean;
  breadthPctAbove50Sma: number;
  adRatio10d: number;
}

export function generateSignals(symbol: string): ModelSignals {
  const isTrending = Math.random() > 0.5;
  const hurst = isTrending ? 0.65 + Math.random() * 0.2 : 0.3 + Math.random() * 0.15;
  
  const vnIndexScore = (Math.random() * 2) - 1; // -1 to 1
  const sectorRS = 0.8 + Math.random() * 0.5; // 0.8 to 1.3
  
  const bullTrapFlag = Math.random() > 0.85;
  const bearTrapFlag = Math.random() > 0.85;
  const fomoSignal = Math.random() > 0.9;
  
  const actionRand = Math.random();
  let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
  let reason = 'Awaiting clear signal';
  
  if (bullTrapFlag) {
    action = 'SELL';
    reason = 'Bull Trap Detected (High Volume Reversal)';
  } else if (fomoSignal) {
    action = 'SELL';
    reason = 'FOMO Signal (Price > 3x ATR from 20-SMA)';
  } else if (actionRand > 0.7 && sectorRS > 1 && hurst > 0.5) {
    action = 'BUY';
    reason = 'Directional Prob > 65% & Sector RS > 1';
  } else if (actionRand < 0.3 || vnIndexScore < -0.5) {
    action = 'SELL';
    reason = 'Lower Bound Breach Risk / Macro Downtrend';
  }
  
  return {
    hurst: Number(hurst.toFixed(2)),
    regime: hurst > 0.5 ? 'Trending' : 'Mean-Reverting',
    vnIndexScore: Number(vnIndexScore.toFixed(2)),
    sectorRS: Number(sectorRS.toFixed(2)),
    vsaRatio: Number((Math.random() * 0.05).toFixed(4)),
    vwapDeviation: Number(((Math.random() - 0.5) * 5).toFixed(2)),
    ceilingDemand: Number((Math.random() * 0.8).toFixed(2)),
    garchSigma: Number((Math.random() * 2 + 1).toFixed(2)),
    directionProb: Number((Math.random() * 100).toFixed(1)),
    t25Risk: Number((Math.random() * 40).toFixed(1)),
    action,
    actionReason: reason,
    usdVndTrend: Number(((Math.random() - 0.5) * 2).toFixed(2)),
    interbankRate: Number((Math.random() * 5).toFixed(2)),
    bullTrapFlag,
    bearTrapFlag,
    fomoSignal,
    breadthPctAbove50Sma: Number((Math.random() * 100).toFixed(1)),
    adRatio10d: Number((Math.random() * 2).toFixed(2)),
  };
}
