export type CatalogRunRow = {
  run_id: string;
  asset: string;
  model_type: string;
  horizon: number;
  feature_set?: string;

  annual_return?: number;
  sharpe?: number;
};

export type Catalog = {
  assets: string[];
  horizons: number[];
  model_types?: string[];
  runs: CatalogRunRow[];
};

export type RunMeta = {
  asset: string;
  model_type: string;
  horizon: number;
  feature_set?: string;
  test_start?: string;
  test_end?: string;

  policy_summary?: string;
  transaction_cost_bps?: number;
};

export type RunMetrics = {
  annual_return?: number;
  annual_volatility?: number;
  sharpe?: number;
  max_drawdown_log?: number;
  avg_turnover_per_year?: number;
  pct_time_in_market?: number;
  hit_rate?: number;
};

export type RunBtRow = {
  date: string;

  cum_ret_net: number;
  cum_ret_gross?: number;
  drawdown_net: number;
  drawdown_gross?: number;

  position_lag: number;
  target_position: number;
  turnover: number;

  net_ret: number;
  strategy_ret: number;

  y_pred: number;
  p_class_0: number;
  p_class_1: number;
  p_class_2: number;
};

export type RunPayload = {
  meta: RunMeta;
  metrics: RunMetrics;
  bt: RunBtRow[];
};

export type BaselineBtRow = {
  date: string;
  cum_ret_net: number;
  drawdown_net: number;
};

export type BaselinePayload = {
  meta?: Record<string, any>;
  bt: BaselineBtRow[];
};

export type SeriesRow = {
  date: string;

  cum_ret_net: number;
  cum_ret_gross?: number;
  drawdown_net: number;
  drawdown_gross?: number;

  position: number;
  target_position: number;
  turnover: number;

  net_ret: number;
  strategy_ret: number;

  p0: number;
  p1: number;
  p2: number;
  y_pred: number;

  base_cum_ret: number | null;
  base_drawdown: number | null;
};

export type HeaderVM = {
  asset: string;
  model: string;
  featureSet: string;
  period: string;
  policy: string;
  costs: string;

  annualReturn?: number;
  vol?: number;
  sharpe?: number;
  mdd?: number;
  turnover?: number;
  tim?: number;
  hit?: number;
};
