"use client";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";
import { useThemeChartColors } from "@/lib/palette";

interface ScatterChartWrapProps {
  data: { x: number; y: number; color?: string }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xRefLines?: number[];
  yRefLines?: number[];
}

export function ScatterChartWrap({
  data, height = 320, xLabel, yLabel, xRefLines, yRefLines,
}: ScatterChartWrapProps) {
  const t = useThemeChartColors();

  const tooltipStyle = {
    fontSize: 12, borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--card, #fff)",
    color: "var(--card-fg, #0a0a0a)",
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ left: 8, right: 16, top: 4, bottom: 28 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={t.grid} />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel ?? "x"}
          tick={{ fontSize: 10, fill: t.muted }}
          tickFormatter={(v: number) => v.toFixed(2)}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fontSize: 11, fill: t.muted } : undefined}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel ?? "y"}
          tick={{ fontSize: 10, fill: t.muted }}
          tickFormatter={(v: number) => v.toFixed(2)}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: t.muted } : undefined}
        />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          contentStyle={tooltipStyle}
          formatter={(v: unknown) => {
            const n = typeof v === "number" ? v : parseFloat(String(v));
            return n.toFixed(4);
          }}
        />
        {xRefLines?.map((x, i) => (
          <ReferenceLine key={"x" + i} x={x} stroke="var(--danger, #dc2626)" strokeDasharray="4 2" />
        ))}
        {yRefLines?.map((y, i) => (
          <ReferenceLine key={"y" + i} y={y} stroke="var(--danger, #dc2626)" strokeDasharray="4 2" />
        ))}
        <Scatter data={data} fillOpacity={0.7}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color ?? "#3b82f6"} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
