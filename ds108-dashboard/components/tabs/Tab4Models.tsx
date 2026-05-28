"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { SimpleBar } from "@/components/charts/BarChartWrap";
import { LineChartWrap } from "@/components/charts/LineChartWrap";
import { InfoIcon } from "@/components/ui/InfoIcon";
import { ExportButton } from "@/components/ui/ExportButton";
import { SkeletonTab } from "@/components/ui/skeleton";
import {
  loadModelResults, loadBacktestResults,
  ModelResults, BacktestResults, fmt3, fmtPct,
} from "@/lib/data";
import { GLOSSARY } from "@/lib/glossary";
import { useChartColors } from "@/lib/palette";

const DATASETS = ["coffee_daily", "coffee_weekly", "corn_daily", "corn_weekly"];
const MODEL_ORDER = ["lgbm", "rf", "lstm", "tcn", "stack_binary", "stack_mc"];

export default function Tab4Models() {
  const [models, setModels] = useState<ModelResults | null>(null);
  const [backtest, setBacktest] = useState<BacktestResults | null>(null);
  const [curveTag, setCurveTag] = useState("coffee_daily");
  const cc = useChartColors();

  useEffect(() => {
    loadModelResults().then(setModels).catch(() => setModels({ rows: [] }));
    loadBacktestResults().then(setBacktest).catch(() => setBacktest({ summary: [], walkforward: [], equity_curves: {} }));
  }, []);

  if (!models || !backtest) return <SkeletonTab rows={24} chartHeight={280} />;

  const orderedRows = [...models.rows].sort((a, b) => {
    const t = a.tag.localeCompare(b.tag);
    if (t !== 0) return t;
    return MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model);
  });

  const sharpeData = backtest.summary.map((r) => ({
    name: r.tag,
    value: r.sharpe,
    color: r.sharpe >= 1 ? "var(--success)" : "var(--warn)",
  }));

  const curveData = backtest.equity_curves[curveTag] ?? [];
  const wfForTag  = backtest.walkforward.filter((r) => r.tag === curveTag);
  const summaryRow = backtest.summary.find((r) => r.tag === curveTag);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelExport: Record<string, any>[] = orderedRows.map((r) => ({
    tag: r.tag, model: r.model, pipeline: r.pipeline,
    test_auc: r.test_auc, test_pr_auc: r.test_pr_auc, test_f1: r.test_f1,
    val_auc: r.val_auc, train_auc: r.train_auc, n_test: r.n_test,
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wfExport: Record<string, any>[] = wfForTag.map((r) => ({
    tag: r.tag, model: r.model, year: r.year,
    sharpe: r.sharpe, mdd: r.mdd, winrate: r.winrate, n_trades: r.n_trades,
  }));

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              Model Metrics — All Datasets
              <InfoIcon tip={GLOSSARY["auc"]} />
            </CardTitle>
            <CardDescription>
              Test AUC ≥ 0.60 highlighted green. Stack rows ensemble base models (LGBM + RF + LSTM + TCN).
            </CardDescription>
          </div>
          <ExportButton rows={modelExport} filename="model_metrics.csv" />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Pipeline</TableHead>
                <TableHead className="text-right">Test AUC</TableHead>
                <TableHead className="text-right">Test PR-AUC</TableHead>
                <TableHead className="text-right">Test F1</TableHead>
                <TableHead className="text-right">Val AUC</TableHead>
                <TableHead className="text-right">Train AUC</TableHead>
                <TableHead className="text-right">N Test</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {orderedRows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center py-8 text-sm" style={{ color: "var(--muted-fg)" }}>
                    No data — run modeling scripts first
                  </TableCell>
                </TableRow>
              )}
              {orderedRows.map((r, i) => (
                <TableRow key={i}>
                  <TableCell className="text-xs">{r.tag}</TableCell>
                  <TableCell className="font-mono text-xs">{r.model}</TableCell>
                  <TableCell><Badge variant="outline" className="text-xs">{r.pipeline}</Badge></TableCell>
                  <TableCell className="text-right">
                    <span
                      className="font-mono text-[13px]"
                      style={{
                        color: r.test_auc >= 0.6 ? "var(--success)" : "var(--fg)",
                        fontWeight: r.test_auc >= 0.6 ? 700 : 500,
                      }}
                    >
                      {fmt3(r.test_auc)}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmt3(r.test_pr_auc)}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmt3(r.test_f1)}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right" style={{ color: "var(--muted-fg)" }}>
                    {fmt3(r.val_auc)}
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right" style={{ color: "var(--muted-fg)" }}>
                    {fmt3(r.train_auc)}
                  </TableCell>
                  <TableCell className="text-[13px] text-right">{r.n_test}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              Sharpe Ratio — Stack per Dataset
              <InfoIcon tip={GLOSSARY["sharpe"]} />
            </CardTitle>
            <CardDescription>Green ≥ 1.0 (acceptable), amber otherwise</CardDescription>
          </CardHeader>
          <CardContent>
            {sharpeData.length > 0 ? (
              <SimpleBar data={sharpeData} height={240} valueFormatter={(v) => v.toFixed(2)} />
            ) : (
              <p className="text-sm py-8 text-center" style={{ color: "var(--muted-fg)" }}>No backtest data</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-start justify-between gap-2">
            <div>
              <CardTitle>Equity Curve</CardTitle>
              <CardDescription>
                {summaryRow
                  ? `Sharpe ${summaryRow.sharpe.toFixed(2)} · MDD ${(summaryRow.mdd * 100).toFixed(1)}% · ${summaryRow.n_trades} trades`
                  : "Select a dataset"}
              </CardDescription>
            </div>
            <Select value={curveTag} onValueChange={(v) => v && setCurveTag(v)}>
              <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
              <SelectContent>
                {DATASETS.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
              </SelectContent>
            </Select>
          </CardHeader>
          <CardContent>
            {curveData.length > 0 ? (
              <LineChartWrap
                data={curveData}
                xKey="date"
                height={260}
                lines={[
                  { dataKey: "model_equity", color: cc.primary,   name: "Model (L/S)" },
                  { dataKey: "bh_equity",    color: "var(--muted-fg)", name: "Buy & Hold" },
                ]}
                referenceLines={[{ y: 1.0, color: "var(--muted-fg)" }]}
                valueFormatter={(v) => v.toFixed(2) + "x"}
              />
            ) : (
              <p className="text-sm py-8 text-center" style={{ color: "var(--muted-fg)" }}>No equity curve for {curveTag}</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              Walk-forward Results — {curveTag}
              <InfoIcon tip={GLOSSARY["walk-forward"]} />
            </CardTitle>
            <CardDescription>Per-year out-of-sample performance, retrained yearly.</CardDescription>
          </div>
          <ExportButton rows={wfExport} filename={`walkforward_${curveTag}.csv`} />
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Year</TableHead>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Sharpe</TableHead>
                <TableHead className="text-right">MDD</TableHead>
                <TableHead className="text-right">Win Rate</TableHead>
                <TableHead className="text-right">N Trades</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {wfForTag.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-sm" style={{ color: "var(--muted-fg)" }}>
                    No walk-forward data for {curveTag}
                  </TableCell>
                </TableRow>
              )}
              {wfForTag.map((r, i) => (
                <TableRow key={i}>
                  <TableCell className="font-medium">{r.year}</TableCell>
                  <TableCell className="font-mono text-xs">{r.model}</TableCell>
                  <TableCell className="text-right">
                    <span className="font-mono text-[13px]" style={{ color: r.sharpe >= 1 ? "var(--success)" : "var(--warn)" }}>
                      {fmt3(r.sharpe)}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right" style={{ color: "var(--danger)" }}>
                    {fmtPct(r.mdd)}
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmtPct(r.winrate)}</TableCell>
                  <TableCell className="text-[13px] text-right">{r.n_trades}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
