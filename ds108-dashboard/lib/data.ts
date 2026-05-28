// TypeScript interfaces and fetch helpers for DS108 static JSON data

export interface DatasetRow {
  name: string;
  tag: string;
  commodity: string;
  freq: string;
  n_total: number;
  n_train: number;
  n_val: number;
  n_test: number;
  base_rate_train: number | null;
  base_rate_val: number | null;
  base_rate_test: number | null;
}
export interface DataQuality {
  datasets: DatasetRow[];
}

export interface FeatureRow {
  feature: string;
  gain: number;
  splits: number;
  group: "market" | "macro" | "weather" | "farming";
}
export type FeatureImportance = Record<string, FeatureRow[]>;

export interface LeakageModule {
  module: string;
  operation: string;
  data_used: string;
  type: string;
  leakage: boolean;
  note: string;
}
export interface AblationEntry {
  coffee_daily: number;
  corn_daily: number;
  delta?: number;
  group: number;
  label: string;
}
export interface LeakageAudit {
  modules: LeakageModule[];
  ablation: Record<string, AblationEntry>;
}

export interface ModelResultRow {
  tag: string;
  model: string;
  pipeline: string;
  test_auc: number;
  test_pr_auc: number | null;
  test_f1: number;
  val_auc: number | null;
  train_auc: number | null;
  best_iter: number | null;
  n_test: number;
}
export interface ModelResults {
  rows: ModelResultRow[];
}

export interface BacktestSummaryRow {
  tag: string;
  model: string;
  sharpe: number;
  mdd: number;
  win_rate: number;
  total_return: number;
  bh_return: number;
  n_trades: number;
  n_test: number;
}
export interface WalkforwardRow {
  tag: string;
  model: string;
  year: string;
  sharpe: number;
  mdd: number;
  winrate: number;
  n_trades: number;
}
export interface EquityPoint {
  date: string;
  model_equity: number;
  bh_equity: number;
}
export interface BacktestResults {
  summary: BacktestSummaryRow[];
  walkforward: WalkforwardRow[];
  equity_curves: Record<string, EquityPoint[]>;
}

export interface HurdleRow {
  tag: string;
  single_r: number;
  half_r: number;
  full_r: number;
  stage2a_r: number;
  n_test: number;
  n_pos_test: number;
  n_neg_test: number;
}
export interface HurdleResults {
  comparison: HurdleRow[];
}

export interface HistSplit {
  n: number;
  flat_pct: number;
  pos_pct: number;
  neg_pct: number;
  mean: number;
  std: number;
  bin_centers: number[];
  counts: number[];
}
export interface ScatterPoint {
  y_true: number;
  hurdle_pred: number;
  is_positive: boolean;
}
export interface HurdleHistDataset {
  full: HistSplit;
  train: HistSplit;
  val: HistSplit;
  test: HistSplit;
  scatter: ScatterPoint[];
}
export type HurdleHistogram = Record<string, HurdleHistDataset>;

async function fetchJson<T>(name: string): Promise<T> {
  const res = await fetch(`/data/${name}`);
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const loadDataQuality = () => fetchJson<DataQuality>("data_quality.json");
export const loadFeatureImportance = () => fetchJson<FeatureImportance>("feature_importance.json");
export const loadLeakageAudit = () => fetchJson<LeakageAudit>("leakage_audit.json");
export const loadModelResults = () => fetchJson<ModelResults>("model_results.json");
export const loadBacktestResults = () => fetchJson<BacktestResults>("backtest_results.json");
export const loadHurdleResults = () => fetchJson<HurdleResults>("hurdle_results.json");
export const loadHurdleHistogram = () => fetchJson<HurdleHistogram>("hurdle_histogram.json");

export function fmt3(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toFixed(3);
}
export function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return (n * 100).toFixed(1) + "%";
}
export function fmtDelta(n: number | null | undefined): string {
  if (n == null) return "—";
  return (n >= 0 ? "+" : "") + n.toFixed(3);
}
