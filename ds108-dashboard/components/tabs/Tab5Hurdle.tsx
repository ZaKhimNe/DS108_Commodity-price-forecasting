"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { SimpleBar } from "@/components/charts/BarChartWrap";
import { ScatterChartWrap } from "@/components/charts/ScatterChartWrap";
import { InfoIcon } from "@/components/ui/InfoIcon";
import { ExportButton } from "@/components/ui/ExportButton";
import { SkeletonTab } from "@/components/ui/skeleton";
import {
  loadHurdleResults, loadHurdleHistogram,
  HurdleResults, HurdleHistogram, HistSplit,
  fmt3,
} from "@/lib/data";
import { GLOSSARY } from "@/lib/glossary";

const DATASETS = ["coffee_daily", "coffee_weekly", "corn_daily", "corn_weekly"];

function HistogramPanel({ splitData }: { splitData: HistSplit }) {
  const chartData = splitData.bin_centers.map((bc, i) => ({
    name: bc.toFixed(3),
    value: splitData.counts[i],
    color: bc < -0.025 ? "var(--danger)" : bc > 0.025 ? "var(--success)" : "var(--muted-fg)",
  }));
  return <SimpleBar data={chartData} height={200} valueFormatter={(v) => String(v)} />;
}

export default function Tab5Hurdle() {
  const [hurdleResults, setHurdleResults] = useState<HurdleResults | null>(null);
  const [histData, setHistData] = useState<HurdleHistogram | null>(null);
  const [selected, setSelected] = useState("coffee_daily");

  useEffect(() => {
    loadHurdleResults().then(setHurdleResults).catch(() => setHurdleResults({ comparison: [] }));
    loadHurdleHistogram().then(setHistData).catch(() => setHistData({}));
  }, []);

  if (!hurdleResults || !histData) return <SkeletonTab rows={4} chartHeight={300} />;

  const ds = histData[selected];
  const scatterData = ds?.scatter.map((p) => ({
    x: p.hurdle_pred,
    y: p.y_true,
    color: p.is_positive ? "var(--success)" : "var(--danger)",
  })) ?? [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const comparisonExport: Record<string, any>[] = hurdleResults.comparison.map((r) => ({
    tag: r.tag, single_r: r.single_r, half_r: r.half_r,
    full_r: r.full_r, stage2a_r: r.stage2a_r, n_test: r.n_test,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              Pearson r — Hurdle vs Single Regression
              <InfoIcon tip={GLOSSARY["hurdle"]} />
            </CardTitle>
            <CardDescription>
              Higher |r| = better. Bold when full hurdle outperforms single regression.
            </CardDescription>
          </div>
          <ExportButton rows={comparisonExport} filename="hurdle_comparison.csv" />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead className="text-right">Single Regression (r)</TableHead>
                <TableHead className="text-right">Half Hurdle (r)</TableHead>
                <TableHead className="text-right">Full Hurdle (r)</TableHead>
                <TableHead className="text-right">Stage2a Pos-only (r)</TableHead>
                <TableHead className="text-right">N Test</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {hurdleResults.comparison.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-sm" style={{ color: "var(--muted-fg)" }}>
                    No data — run hurdle model scripts first
                  </TableCell>
                </TableRow>
              )}
              {hurdleResults.comparison.map((r) => (
                <TableRow key={r.tag}>
                  <TableCell className="font-medium text-[13px]">{r.tag}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmt3(r.single_r)}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmt3(r.half_r)}</TableCell>
                  <TableCell className="text-right">
                    <span
                      className="font-mono text-[13px]"
                      style={{
                        color: Math.abs(r.full_r) > Math.abs(r.single_r) ? "var(--success)" : "var(--fg)",
                        fontWeight: Math.abs(r.full_r) > Math.abs(r.single_r) ? 700 : 500,
                      }}
                    >
                      {fmt3(r.full_r)}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmt3(r.stage2a_r)}</TableCell>
                  <TableCell className="text-[13px] text-right">{r.n_test}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="flex items-center gap-4">
        <span className="text-sm font-medium" style={{ color: "var(--fg)" }}>Dataset:</span>
        <Select value={selected} onValueChange={(v) => v && setSelected(v)}>
          <SelectTrigger className="w-48"><SelectValue /></SelectTrigger>
          <SelectContent>
            {DATASETS.map((d) => <SelectItem key={d} value={d}>{d}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {ds && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Return Distribution — Full Dataset</CardTitle>
              <CardDescription>
                neg={(( ds.full.neg_pct ?? 0) * 100).toFixed(1)}% (red) | flat={(( ds.full.flat_pct ?? 0) * 100).toFixed(1)}% (gray) | pos={(( ds.full.pos_pct ?? 0) * 100).toFixed(1)}% (green) | n={ds.full.n}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <HistogramPanel splitData={ds.full} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Return Distribution — Test Set</CardTitle>
              <CardDescription>
                neg={(( ds.test.neg_pct ?? 0) * 100).toFixed(1)}% | flat={(( ds.test.flat_pct ?? 0) * 100).toFixed(1)}% | pos={(( ds.test.pos_pct ?? 0) * 100).toFixed(1)}% | n={ds.test.n}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <HistogramPanel splitData={ds.test} />
            </CardContent>
          </Card>

          <Card className="xl:col-span-2">
            <CardHeader>
              <CardTitle>Scatter: True Return vs Full Hurdle Prediction (Test, n=300 sample)</CardTitle>
              <CardDescription>Green = positive return · Red = negative return</CardDescription>
            </CardHeader>
            <CardContent>
              <ScatterChartWrap
                data={scatterData}
                height={320}
                xLabel="Hurdle Prediction"
                yLabel="True Return"
                yRefLines={[-0.025, 0.025]}
                xRefLines={[0]}
              />
            </CardContent>
          </Card>
        </div>
      )}

      {!ds && (
        <p className="text-sm py-8 text-center" style={{ color: "var(--muted-fg)" }}>
          No histogram data for {selected}
        </p>
      )}
    </div>
  );
}
