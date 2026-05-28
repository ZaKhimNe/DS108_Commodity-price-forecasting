"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from "recharts";
import type { ValueType } from "recharts/types/component/DefaultTooltipContent";
import { useChartColors, useThemeChartColors } from "@/lib/palette";

interface SimpleBarProps {
  data: { name: string; value: number; color?: string; group?: string }[];
  height?: number;
  horizontal?: boolean;
  valueFormatter?: (v: number) => string;
}

export function SimpleBar({ data, height = 300, horizontal = false, valueFormatter }: SimpleBarProps) {
  const cc = useChartColors();
  const t = useThemeChartColors();

  const fmt = (v: ValueType | undefined) => {
    const n = typeof v === "number" ? v : parseFloat(String(v ?? 0));
    return valueFormatter ? valueFormatter(n) : n.toFixed(3);
  };

  const tooltipStyle = {
    fontSize: 12, borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--card, #fff)",
    color: "var(--card-fg, #0a0a0a)",
  };

  if (horizontal) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={t.grid} />
          <XAxis type="number" tick={{ fontSize: 11, fill: t.muted }} tickFormatter={valueFormatter} />
          <YAxis dataKey="name" type="category" width={180} tick={{ fontSize: 11, fill: t.muted }} />
          <Tooltip formatter={fmt} contentStyle={tooltipStyle} cursor={{ fill: "rgba(127,127,127,.06)" }} />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.color ?? (d.group ? cc[d.group as keyof typeof cc] as string : undefined) ?? cc.primary}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={t.grid} />
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: t.muted }} angle={-30} textAnchor="end" interval={0} />
        <YAxis tick={{ fontSize: 11, fill: t.muted }} tickFormatter={valueFormatter} />
        <Tooltip formatter={fmt} contentStyle={tooltipStyle} cursor={{ fill: "rgba(127,127,127,.06)" }} />
        <Bar dataKey="value" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={d.color ?? (d.group ? cc[d.group as keyof typeof cc] as string : undefined) ?? cc.primary}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

interface GroupedBarProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any[];
  keys: { dataKey: string; color?: string; name?: string }[];
  xKey: string;
  height?: number;
  valueFormatter?: (v: number) => string;
}

export function GroupedBar({ data, keys, xKey, height = 280, valueFormatter }: GroupedBarProps) {
  const cc = useChartColors();
  const t = useThemeChartColors();
  const paletteValues = Object.values(cc);

  const fmt = (v: ValueType | undefined) => {
    const n = typeof v === "number" ? v : parseFloat(String(v ?? 0));
    return valueFormatter ? valueFormatter(n) : n.toFixed(3);
  };

  const tooltipStyle = {
    fontSize: 12, borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--card, #fff)",
    color: "var(--card-fg, #0a0a0a)",
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={t.grid} />
        <XAxis dataKey={xKey} tick={{ fontSize: 11, fill: t.muted }} angle={-30} textAnchor="end" interval={0} />
        <YAxis tick={{ fontSize: 11, fill: t.muted }} tickFormatter={valueFormatter} />
        <Tooltip formatter={fmt} contentStyle={tooltipStyle} cursor={{ fill: "rgba(127,127,127,.06)" }} />
        <Legend wrapperStyle={{ fontSize: 12, color: t.fg }} />
        {keys.map((k, i) => (
          <Bar
            key={k.dataKey}
            dataKey={k.dataKey}
            fill={k.color ?? paletteValues[i] as string ?? cc.primary}
            name={k.name ?? k.dataKey}
            radius={[2, 2, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
