"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleBar } from "@/components/charts/BarChartWrap";
import { InfoIcon } from "@/components/ui/InfoIcon";
import { ExportButton } from "@/components/ui/ExportButton";
import { SkeletonTab } from "@/components/ui/skeleton";
import { loadFeatureImportance, FeatureImportance } from "@/lib/data";
import { GLOSSARY } from "@/lib/glossary";
import { useChartColors } from "@/lib/palette";

const DATASETS = ["coffee_daily", "coffee_weekly", "corn_daily", "corn_weekly"];

export default function Tab2Features() {
  const [data, setData] = useState<FeatureImportance | null>(null);
  const [selected, setSelected] = useState("coffee_daily");
  const cc = useChartColors();

  useEffect(() => {
    loadFeatureImportance().then(setData).catch(() => setData({}));
  }, []);

  if (!data) return <SkeletonTab rows={20} chartHeight={520} />;

  const rows = (data[selected] ?? []).slice(0, 20);
  const chartData = [...rows]
    .sort((a, b) => a.gain - b.gain)
    .map((r) => ({ name: r.feature, value: r.gain, group: r.group }));

  const groups = ["market", "macro", "weather", "farming"] as const;
  const groupCounts = groups.map((g) => ({
    name: g,
    value: (data[selected] ?? []).filter((r) => r.group === g).length,
    color: cc[g],
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const exportRows: Record<string, any>[] = rows.map((r) => ({
    feature: r.feature, group: r.group, gain: r.gain, splits: r.splits,
  }));

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-[13px] font-medium" style={{ color: "var(--fg)" }}>Dataset:</span>
        <Select value={selected} onValueChange={(v) => v && setSelected(v)}>
          <SelectTrigger className="w-48"><SelectValue /></SelectTrigger>
          <SelectContent>
            {DATASETS.map((d) => <SelectItem key={d} value={d}>{d}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <Card className="xl:col-span-2">
          <CardHeader className="flex flex-row items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-1.5">
                Top 20 Features by Gain (LGBM)
                <InfoIcon tip={GLOSSARY["auc"]} />
              </CardTitle>
              <CardDescription>Colored by feature group — {selected}</CardDescription>
            </div>
            <ExportButton rows={exportRows} filename={`features_${selected}.csv`} />
          </CardHeader>
          <CardContent>
            {chartData.length > 0 ? (
              <SimpleBar data={chartData} height={520} horizontal />
            ) : (
              <p className="text-sm py-8 text-center" style={{ color: "var(--muted-fg)" }}>No data for {selected}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Feature Groups</CardTitle>
            <CardDescription>Composition of selected feature set</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2.5">
              {groupCounts.map((g) => (
                <div key={g.name} className="flex items-center justify-between text-[13px]">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: g.color }} />
                    <span className="capitalize" style={{ color: "var(--fg)" }}>{g.name}</span>
                  </div>
                  <span className="font-mono font-medium tabular-nums" style={{ color: "var(--fg)" }}>
                    {g.value}
                  </span>
                </div>
              ))}
            </div>
            <div
              className="mt-4 pt-4 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <SimpleBar data={groupCounts} height={180} valueFormatter={(v) => String(v)} />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
