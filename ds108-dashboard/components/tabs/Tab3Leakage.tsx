"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { InfoIcon } from "@/components/ui/InfoIcon";
import { ExportButton } from "@/components/ui/ExportButton";
import { SkeletonTab } from "@/components/ui/skeleton";
import { loadLeakageAudit, LeakageAudit, AblationEntry, fmt3 } from "@/lib/data";
import { GLOSSARY } from "@/lib/glossary";

const GROUP_LABELS: Record<number, string> = {
  1: "Nhóm 1 — Stacking Ensemble",
  2: "Nhóm 2 — LGBM Standalone",
  3: "Nhóm 3 — Leakage Experiments",
};

export default function Tab3Leakage() {
  const [data, setData] = useState<LeakageAudit | null>(null);

  useEffect(() => {
    loadLeakageAudit().then(setData).catch(() => setData({ modules: [], ablation: {} }));
  }, []);

  if (!data) return <SkeletonTab rows={10} chartHeight={0} />;

  const ablationByGroup = [1, 2, 3].map((g) => ({
    group: g,
    label: GROUP_LABELS[g],
    rows: Object.entries(data.ablation).filter(([, v]) => (v as AblationEntry).group === g),
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const moduleExport: Record<string, any>[] = data.modules.map((m) => ({
    module: m.module, operation: m.operation, data_used: m.data_used,
    type: m.type, leakage: m.leakage, note: m.note,
  }));

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              Leakage Audit — Pipeline Modules
              <InfoIcon tip={GLOSSARY["leakage"]} />
            </CardTitle>
            <CardDescription>
              Every module reviewed for look-ahead, target leakage, or improper scaling/resampling.
            </CardDescription>
          </div>
          <ExportButton rows={moduleExport} filename="leakage_audit_modules.csv" />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Module</TableHead>
                <TableHead>Operation</TableHead>
                <TableHead>Data Used</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Rò rỉ</TableHead>
                <TableHead>Note</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.modules.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-sm" style={{ color: "var(--muted-fg)" }}>
                    No data — run export script to generate leakage_audit.json
                  </TableCell>
                </TableRow>
              )}
              {data.modules.map((m, i) => (
                <TableRow key={i}>
                  <TableCell className="font-medium text-xs">{m.module}</TableCell>
                  <TableCell className="text-xs">{m.operation}</TableCell>
                  <TableCell className="text-xs font-mono" style={{ color: "var(--muted-fg)" }}>{m.data_used}</TableCell>
                  <TableCell className="text-xs">{m.type}</TableCell>
                  <TableCell>
                    <Badge variant={m.leakage ? "destructive" : "outline"} className={m.leakage ? "" : "text-emerald-600 border-emerald-500"}>
                      {m.leakage ? "Có" : "Không"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs" style={{ color: "var(--muted-fg)" }}>{m.note}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            Ablation Study — AUC Comparison
            <InfoIcon tip={GLOSSARY["ablation"]} />
          </CardTitle>
          <CardDescription>
            Δ AUC vs full config. Nhóm 3 cố tình inject leakage để verify detection.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Cấu hình</TableHead>
                <TableHead className="text-right">Coffee Daily</TableHead>
                <TableHead className="text-right">Corn Daily</TableHead>
                <TableHead className="text-right">Δ AUC</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ablationByGroup.map(({ label, rows }) => (
                <TableBody key={label}>
                  <TableRow>
                    <TableCell
                      colSpan={4}
                      className="font-semibold text-xs italic py-2"
                      style={{ background: "var(--row-hover)" }}
                    >
                      {label}
                    </TableCell>
                  </TableRow>
                  {rows.map(([key, val]) => {
                    const v = val as AblationEntry;
                    return (
                      <TableRow key={key}>
                        <TableCell className="pl-8 text-[13px]">{v.label}</TableCell>
                        <TableCell className="font-mono text-[13px] text-right">{fmt3(v.coffee_daily)}</TableCell>
                        <TableCell className="font-mono text-[13px] text-right">{fmt3(v.corn_daily)}</TableCell>
                        <TableCell className="text-right">
                          {v.delta != null ? (
                            <span
                              className="font-mono text-[13px] font-semibold"
                              style={{
                                color: (v.delta ?? 0) > 0.05
                                  ? "var(--danger)"
                                  : (v.delta ?? 0) < -0.005
                                  ? "var(--warn)"
                                  : "var(--muted-fg)",
                              }}
                            >
                              {(v.delta ?? 0) > 0 ? "+" : ""}{fmt3(v.delta)}
                            </span>
                          ) : (
                            <span style={{ color: "var(--muted-fg)" }}>—</span>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
