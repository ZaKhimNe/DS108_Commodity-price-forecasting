"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { ValueType } from "recharts/types/component/DefaultTooltipContent";
import { useChartColors, useThemeChartColors } from "@/lib/palette";

interface LineChartWrapProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any[];
  lines: { dataKey: string; color?: string; name?: string; dot?: boolean }[];
  xKey: string;
  height?: number;
  referenceLines?: { y: number; label?: string; color?: string }[];
  valueFormatter?: (v: number) => string;
}

export function LineChartWrap({
  data, lines, xKey, height = 320, referenceLines, valueFormatter,
}: LineChartWrapProps) {
  const cc = useChartColors();
  const t = useThemeChartColors();
  const paletteValues = Object.values(cc);

  const fmt = (v: ValueType | undefined) => {
    const n = typeof v === "number" ? v : parseFloat(String(v ?? 0));
    return valueFormatter ? valueFormatter(n) : n.toFixed(4);
  };

  const tooltipStyle = {
    fontSize: 12, borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--card, #fff)",
    color: "var(--card-fg, #0a0a0a)",
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={t.grid} />
        <XAxis
          dataKey={xKey}
          tick={{ fontSize: 10, fill: t.muted }}
          interval={Math.max(0, Math.floor(data.length / 6))}
        />
        <YAxis tick={{ fontSize: 11, fill: t.muted }} tickFormatter={valueFormatter} />
        <Tooltip
          formatter={fmt}
          contentStyle={tooltipStyle}
          labelFormatter={(l) => String(l)}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: t.fg }} />
        {referenceLines?.map((rl, i) => (
          <ReferenceLine
            key={i}
            y={rl.y}
            stroke={rl.color ?? t.muted}
            strokeDasharray="4 2"
            label={rl.label}
          />
        ))}
        {lines.map((l, i) => (
          <Line
            key={l.dataKey}
            type="monotone"
            dataKey={l.dataKey}
            stroke={l.color ?? paletteValues[i] as string ?? cc.primary}
            name={l.name ?? l.dataKey}
            dot={l.dot ?? false}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
