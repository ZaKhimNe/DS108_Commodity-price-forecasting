"use client";
import { createContext, useContext, useEffect, useState } from "react";

export const PALETTES = {
  default: {
    market: "#3b82f6", macro: "#f59e0b", weather: "#10b981", farming: "#8b5cf6",
    primary: "#3b82f6", secondary: "#f59e0b", tertiary: "#10b981", quaternary: "#8b5cf6",
  },
  mono: {
    market: "#171717", macro: "#525252", weather: "#a3a3a3", farming: "#d4d4d4",
    primary: "#171717", secondary: "#525252", tertiary: "#a3a3a3", quaternary: "#d4d4d4",
  },
  jewel: {
    market: "#0f766e", macro: "#0e7490", weather: "#1d4ed8", farming: "#7e22ce",
    primary: "#0f766e", secondary: "#0e7490", tertiary: "#1d4ed8", quaternary: "#7e22ce",
  },
  neon: {
    market: "#fb923c", macro: "#22d3ee", weather: "#a3e635", farming: "#f472b6",
    primary: "#fb923c", secondary: "#22d3ee", tertiary: "#a3e635", quaternary: "#f472b6",
  },
};

export type PaletteName = keyof typeof PALETTES;

export const ChartPaletteCtx = createContext<PaletteName>("default");

export function useChartColors() {
  const name = useContext(ChartPaletteCtx);
  return PALETTES[name] ?? PALETTES.default;
}

export function useThemeChartColors() {
  const [vars, setVars] = useState({ fg: "#0a0a0a", muted: "#737373", grid: "#e5e5e5" });
  useEffect(() => {
    const update = () => {
      const cs = getComputedStyle(document.documentElement);
      setVars({
        fg:    cs.getPropertyValue("--fg").trim()          || "#0a0a0a",
        muted: cs.getPropertyValue("--muted-fg").trim()    || "#737373",
        grid:  cs.getPropertyValue("--chart-grid").trim()  || "#e5e5e5",
      });
    };
    update();
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return vars;
}
