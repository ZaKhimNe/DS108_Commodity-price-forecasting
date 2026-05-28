"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { GroupedBar } from "@/components/charts/BarChartWrap";
import { InfoIcon } from "@/components/ui/InfoIcon";
import { ExportButton } from "@/components/ui/ExportButton";
import { SkeletonTab } from "@/components/ui/skeleton";
import { loadDataQuality, DataQuality, fmtPct } from "@/lib/data";
import { GLOSSARY } from "@/lib/glossary";

export default function Tab1DataQuality() {
  const [data, setData] = useState<DataQuality | null>(null);

  useEffect(() => {
    loadDataQuality().then(setData).catch(() => setData({ datasets: [] }));
  }, []);

  if (!data) return <SkeletonTab rows={4} chartHeight={280} />;

  const chartData = data.datasets.map((d) => ({
    name: d.name,
    Train: d.base_rate_train ?? 0,
    Val:   d.base_rate_val   ?? 0,
    Test:  d.base_rate_test  ?? 0,
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const exportRows: Record<string, any>[] = data.datasets.map((d) => ({
    name: d.name, commodity: d.commodity, freq: d.freq,
    n_total: d.n_total, n_train: d.n_train, n_val: d.n_val, n_test: d.n_test,
    base_rate_train: d.base_rate_train, base_rate_val: d.base_rate_val, base_rate_test: d.base_rate_test,
  }));

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              Split Statistics
              <InfoIcon tip={GLOSSARY["base rate"]} />
            </CardTitle>
            <CardDescription>Chronological split per dataset (train 70% / val 10% / test 20%)</CardDescription>
          </div>
          <ExportButton rows={exportRows} filename="data_quality_splits.csv" />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Freq</TableHead>
                <TableHead className="text-right">N Total</TableHead>
                <TableHead className="text-right">N Train</TableHead>
                <TableHead className="text-right">N Val</TableHead>
                <TableHead className="text-right">N Test</TableHead>
                <TableHead className="text-right">Base Rate Train</TableHead>
                <TableHead className="text-right">Base Rate Val</TableHead>
                <TableHead className="text-right">Base Rate Test</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.datasets.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center py-8 text-sm" style={{ color: "var(--muted-fg)" }}>
                    No data — run export script to generate data_quality.json
                  </TableCell>
                </TableRow>
              )}
              {data.datasets.map((d) => (
                <TableRow key={d.name}>
                  <TableCell className="font-medium text-sm">{d.name}</TableCell>
                  <TableCell>
                    <Badge variant={d.freq === "daily" ? "default" : "secondary"}>{d.freq}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{d.n_total.toLocaleString()}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{d.n_train.toLocaleString()}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{d.n_val.toLocaleString()}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{d.n_test.toLocaleString()}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmtPct(d.base_rate_train)}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmtPct(d.base_rate_val)}</TableCell>
                  <TableCell className="font-mono text-[13px] text-right">{fmtPct(d.base_rate_test)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Base Rate per Split</CardTitle>
          <CardDescription>Positive-class proportion across splits. Stable bars = no distribution drift.</CardDescription>
        </CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <GroupedBar
              data={chartData}
              xKey="name"
              height={280}
              keys={[
                { dataKey: "Train", name: "Train" },
                { dataKey: "Val",   name: "Val"   },
                { dataKey: "Test",  name: "Test"  },
              ]}
              valueFormatter={(v) => fmtPct(v)}
            />
          ) : (
            <p className="text-sm py-8 text-center" style={{ color: "var(--muted-fg)" }}>No chart data</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
