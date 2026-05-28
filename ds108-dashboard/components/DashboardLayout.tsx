"use client";
import { useEffect, useRef, useState } from "react";
import { ChartPaletteCtx, PaletteName } from "@/lib/palette";
import { GLOSSARY } from "@/lib/glossary";
import {
  loadModelResults, loadBacktestResults, loadLeakageAudit,
  ModelResults, BacktestResults, LeakageAudit, fmt3, fmtPct,
} from "@/lib/data";
import Tab1DataQuality from "@/components/tabs/Tab1DataQuality";
import Tab2Features from "@/components/tabs/Tab2Features";
import Tab3Leakage from "@/components/tabs/Tab3Leakage";
import Tab4Models from "@/components/tabs/Tab4Models";
import Tab5Hurdle from "@/components/tabs/Tab5Hurdle";

// ── Preferences types ─────────────────────────────────────────────────────
type FontOption  = "inter" | "outfit" | "ibm-plex" | "dm-sans";
type DensityOpt  = "compact" | "regular" | "comfy";

interface Prefs {
  dark:    boolean;
  palette: PaletteName;
  font:    FontOption;
  density: DensityOpt;
  accent:  string;
}

const DEFAULT_PREFS: Prefs = {
  dark:    false,
  palette: "default",
  font:    "inter",
  density: "regular",
  accent:  "#3b82f6",
};

const ACCENT_PRESETS = ["#3b82f6", "#0f766e", "#7c3aed", "#dc2626", "#f59e0b"];
const FONT_OPTIONS: { value: FontOption; label: string; sample: string }[] = [
  { value: "inter",    label: "Inter",          sample: "Aa" },
  { value: "outfit",   label: "Outfit",         sample: "Aa" },
  { value: "ibm-plex", label: "IBM Plex Sans",  sample: "Aa" },
  { value: "dm-sans",  label: "DM Sans",        sample: "Aa" },
];
const PALETTE_OPTIONS: { value: PaletteName; label: string; colors: string[] }[] = [
  { value: "default", label: "Default",  colors: ["#3b82f6","#f59e0b","#10b981","#8b5cf6"] },
  { value: "mono",    label: "Mono",     colors: ["#171717","#525252","#a3a3a3","#d4d4d4"] },
  { value: "jewel",   label: "Jewel",    colors: ["#0f766e","#0e7490","#1d4ed8","#7e22ce"] },
  { value: "neon",    label: "Neon",     colors: ["#fb923c","#22d3ee","#a3e635","#f472b6"] },
];
const DENSITY_OPTIONS: { value: DensityOpt; label: string }[] = [
  { value: "compact", label: "Compact" },
  { value: "regular", label: "Regular" },
  { value: "comfy",   label: "Comfy"   },
];

// ── SVG icon set ──────────────────────────────────────────────────────────
function NavIcon({ name }: { name: string }) {
  const p: React.SVGProps<SVGSVGElement> = {
    width: 16, height: 16, viewBox: "0 0 24 24", fill: "none",
    stroke: "currentColor", strokeWidth: 2,
    strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
  };
  switch (name) {
    case "database": return <svg {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>;
    case "list":     return <svg {...p}><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>;
    case "shield":   return <svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>;
    case "chart":    return <svg {...p}><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>;
    case "split":    return <svg {...p}><path d="M6 3v12"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="6" r="3"/><path d="M18 9v3a3 3 0 0 1-3 3H9"/></svg>;
    case "sun":      return <svg {...p}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>;
    case "moon":     return <svg {...p}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>;
    case "menu":     return <svg {...p}><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>;
    case "settings": return <svg {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>;
    case "close":    return <svg {...p}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;
    default: return null;
  }
}

// ── Nav definition ─────────────────────────────────────────────────────────
const NAV = [
  { id: "tab1", label: "Data Quality",  desc: "Splits & base rates",  icon: "database" },
  { id: "tab2", label: "Features",      desc: "Importance & groups",  icon: "list"     },
  { id: "tab3", label: "Leakage Audit", desc: "Modules & ablation",   icon: "shield"   },
  { id: "tab4", label: "Model Results", desc: "Metrics & backtest",   icon: "chart"    },
  { id: "tab5", label: "Hurdle Model",  desc: "Two-stage regression", icon: "split"    },
];

// ── Preferences Panel ──────────────────────────────────────────────────────
function PreferencesPanel({
  prefs,
  onChange,
  onClose,
}: {
  prefs: Prefs;
  onChange: <K extends keyof Prefs>(key: K, val: Prefs[K]) => void;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  const section = (label: string) => (
    <div className="text-[10px] uppercase tracking-widest font-semibold mt-5 mb-2 first:mt-0" style={{ color: "var(--muted-fg)" }}>
      {label}
    </div>
  );

  const iconBtn = (active: boolean) =>
    `w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 transition-all ${
      active ? "border-[var(--accent-ds)]" : "border-[var(--muted-fg)]"
    }`;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40" style={{ background: "rgba(0,0,0,.15)" }} />

      {/* Panel */}
      <div
        ref={panelRef}
        className="fixed right-0 top-0 h-screen z-50 flex flex-col overflow-y-auto"
        style={{
          width: 300,
          background: "var(--card, #fff)",
          borderLeft: "1px solid var(--border)",
          boxShadow: "-8px 0 32px rgba(0,0,0,.12)",
        }}
      >
        {/* Header */}
        <div
          className="h-16 px-5 flex items-center justify-between shrink-0 border-b"
          style={{ borderColor: "var(--border)" }}
        >
          <span className="text-[14px] font-semibold" style={{ color: "var(--fg)" }}>Preferences</span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg transition-colors"
            style={{ color: "var(--muted-fg)" }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--fg)"; (e.currentTarget as HTMLElement).style.background = "var(--row-hover)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--muted-fg)"; (e.currentTarget as HTMLElement).style.background = ""; }}
          >
            <NavIcon name="close" />
          </button>
        </div>

        <div className="p-5 flex-1">
          {/* ── Appearance ── */}
          {section("Appearance")}

          {/* Dark mode */}
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px]" style={{ color: "var(--fg)" }}>Dark mode</span>
            <button
              onClick={() => onChange("dark", !prefs.dark)}
              className="relative w-10 h-5 rounded-full transition-colors shrink-0"
              style={{ background: prefs.dark ? "var(--accent-ds)" : "var(--muted-fg)" }}
            >
              <span
                className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all"
                style={{ left: prefs.dark ? "calc(100% - 18px)" : "2px" }}
              />
            </button>
          </div>

          {/* Accent color */}
          <div className="mb-1">
            <span className="text-[13px]" style={{ color: "var(--fg)" }}>Accent color</span>
            <div className="flex gap-2 mt-2">
              {ACCENT_PRESETS.map((c) => (
                <button
                  key={c}
                  onClick={() => onChange("accent", c)}
                  className="w-7 h-7 rounded-full border-2 transition-all"
                  style={{
                    background: c,
                    borderColor: prefs.accent === c ? c : "transparent",
                    boxShadow: prefs.accent === c ? `0 0 0 2px var(--card), 0 0 0 4px ${c}` : "none",
                  }}
                />
              ))}
            </div>
          </div>

          {/* ── Typography ── */}
          {section("Typography")}
          <div className="space-y-1.5">
            {FONT_OPTIONS.map((f) => (
              <button
                key={f.value}
                onClick={() => onChange("font", f.value)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors"
                style={{
                  background: prefs.font === f.value ? "var(--muted)" : "transparent",
                  border: `1px solid ${prefs.font === f.value ? "var(--accent-ds)" : "transparent"}`,
                }}
                onMouseEnter={(e) => { if (prefs.font !== f.value) (e.currentTarget as HTMLElement).style.background = "var(--row-hover)"; }}
                onMouseLeave={(e) => { if (prefs.font !== f.value) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span
                  className="w-8 h-8 rounded-md flex items-center justify-center text-[15px] font-semibold shrink-0"
                  style={{
                    fontFamily: `var(--font-${f.value === "ibm-plex" ? "ibm-plex" : f.value})`,
                    background: prefs.font === f.value ? "var(--accent-ds)" : "var(--muted)",
                    color: prefs.font === f.value ? "#fff" : "var(--fg)",
                  }}
                >
                  Aa
                </span>
                <div className="flex flex-col min-w-0">
                  <span
                    className="text-[13px] font-medium leading-tight"
                    style={{
                      fontFamily: `var(--font-${f.value === "ibm-plex" ? "ibm-plex" : f.value})`,
                      color: "var(--fg)",
                    }}
                  >
                    {f.label}
                  </span>
                </div>
                {prefs.font === f.value && (
                  <span className="ml-auto text-[11px] font-semibold" style={{ color: "var(--accent-ds)" }}>✓</span>
                )}
              </button>
            ))}
          </div>

          {/* ── Charts ── */}
          {section("Chart palette")}
          <div className="space-y-1.5">
            {PALETTE_OPTIONS.map((p) => (
              <button
                key={p.value}
                onClick={() => onChange("palette", p.value)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors"
                style={{
                  background: prefs.palette === p.value ? "var(--muted)" : "transparent",
                  border: `1px solid ${prefs.palette === p.value ? "var(--accent-ds)" : "transparent"}`,
                }}
                onMouseEnter={(e) => { if (prefs.palette !== p.value) (e.currentTarget as HTMLElement).style.background = "var(--row-hover)"; }}
                onMouseLeave={(e) => { if (prefs.palette !== p.value) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <div className="flex gap-0.5 shrink-0">
                  {p.colors.map((c, i) => (
                    <div key={i} className="w-4 h-4 rounded-sm" style={{ background: c }} />
                  ))}
                </div>
                <span className="text-[13px]" style={{ color: "var(--fg)" }}>{p.label}</span>
                {prefs.palette === p.value && (
                  <span className="ml-auto text-[11px] font-semibold" style={{ color: "var(--accent-ds)" }}>✓</span>
                )}
              </button>
            ))}
          </div>

          {/* ── Layout ── */}
          {section("Layout density")}
          <div className="flex gap-2">
            {DENSITY_OPTIONS.map((d) => (
              <button
                key={d.value}
                onClick={() => onChange("density", d.value)}
                className="flex-1 py-2 rounded-lg text-[12px] font-medium transition-colors border"
                style={{
                  background: prefs.density === d.value ? "var(--accent-ds)" : "transparent",
                  color: prefs.density === d.value ? "#fff" : "var(--muted-fg)",
                  borderColor: prefs.density === d.value ? "var(--accent-ds)" : "var(--border)",
                }}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        <div
          className="p-4 text-[11px] text-center border-t"
          style={{ color: "var(--muted-fg)", borderColor: "var(--border)" }}
        >
          Settings saved automatically
        </div>
      </div>
    </>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────
function Sidebar({ active, onSelect, collapsed, accent }: {
  active: string;
  onSelect: (id: string) => void;
  collapsed: boolean;
  accent: string;
}) {
  return (
    <aside
      className="shrink-0 sticky top-0 h-screen border-r flex flex-col z-30 transition-all duration-200"
      style={{
        width: collapsed ? 64 : 240,
        background: "var(--card, #fff)",
        borderColor: "var(--border, rgba(0,0,0,.08))",
      }}
    >
      <div
        className="h-16 px-4 flex items-center gap-2.5 border-b shrink-0"
        style={{ borderColor: "var(--border, rgba(0,0,0,.08))" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-white shrink-0 font-semibold text-[13px] tracking-tight"
          style={{ background: accent }}
        >
          DS
        </div>
        {!collapsed && (
          <div className="flex flex-col min-w-0">
            <span className="text-[13px] font-semibold tracking-tight truncate" style={{ color: "var(--fg)" }}>DS108</span>
            <span className="text-[11px] truncate" style={{ color: "var(--muted-fg)" }}>Pipeline Results</span>
          </div>
        )}
      </div>

      <nav className="p-2 flex flex-col gap-0.5 flex-1 overflow-y-auto">
        {NAV.map((n, i) => {
          const isActive = active === n.id;
          return (
            <button
              key={n.id}
              onClick={() => onSelect(n.id)}
              title={collapsed ? n.label : undefined}
              className="w-full flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors"
              style={{
                background: isActive ? "var(--muted, #f5f5f4)" : "transparent",
                color: isActive ? "var(--fg)" : "var(--muted-fg)",
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.background = "var(--row-hover, rgba(0,0,0,.025))";
                  (e.currentTarget as HTMLElement).style.color = "var(--fg)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                  (e.currentTarget as HTMLElement).style.color = "var(--muted-fg)";
                }
              }}
            >
              <span className="shrink-0" style={{ color: isActive ? accent : undefined }}>
                <NavIcon name={n.icon} />
              </span>
              {!collapsed && (
                <div className="flex flex-col min-w-0 flex-1">
                  <span className="text-[13px] font-medium leading-tight truncate">{i + 1}. {n.label}</span>
                  <span className="text-[11px] truncate" style={{ color: "var(--muted-fg)" }}>{n.desc}</span>
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {!collapsed && (
        <div
          className="p-3 border-t text-[10px] leading-relaxed shrink-0"
          style={{ borderColor: "var(--border, rgba(0,0,0,.08))", color: "var(--muted-fg)" }}
        >
          <div className="uppercase tracking-wider font-medium mb-1">Pipeline</div>
          <div>Stack ensemble · LGBM + RF + LSTM + TCN</div>
          <div>Walk-forward 2022–2024</div>
        </div>
      )}
    </aside>
  );
}

// ── TopBar ──────────────────────────────────────────────────────────────────
function TopBar({ dark, onToggleDark, onToggleSidebar, onOpenPrefs }: {
  dark: boolean;
  onToggleDark: () => void;
  onToggleSidebar: () => void;
  onOpenPrefs: () => void;
}) {
  const iconBtnStyle = { color: "var(--muted-fg)" } as React.CSSProperties;
  const hoverIn  = (e: React.MouseEvent) => { (e.currentTarget as HTMLElement).style.color = "var(--fg)"; (e.currentTarget as HTMLElement).style.background = "var(--row-hover)"; };
  const hoverOut = (e: React.MouseEvent) => { (e.currentTarget as HTMLElement).style.color = "var(--muted-fg)"; (e.currentTarget as HTMLElement).style.background = ""; };

  return (
    <header
      className="sticky top-0 z-20 h-16 border-b px-4 flex items-center gap-3 shrink-0"
      style={{ background: "var(--bg, #fafaf9)", borderColor: "var(--border, rgba(0,0,0,.08))" }}
    >
      <button onClick={onToggleSidebar} className="p-2 rounded-lg transition-colors" style={iconBtnStyle} onMouseEnter={hoverIn} onMouseLeave={hoverOut} aria-label="Toggle sidebar">
        <NavIcon name="menu" />
      </button>

      <div className="hidden md:flex flex-col">
        <h1 className="text-[15px] font-semibold tracking-tight leading-tight" style={{ color: "var(--fg)" }}>
          Pipeline Results Dashboard
        </h1>
        <p className="text-[11px] leading-tight" style={{ color: "var(--muted-fg)" }}>
          Coffee &amp; Corn Futures · Walk-forward 2022–2024
        </p>
      </div>

      <div className="flex-1" />

      <button onClick={onToggleDark} className="p-2 rounded-lg border transition-colors" style={{ ...iconBtnStyle, borderColor: "var(--border)" }} onMouseEnter={hoverIn} onMouseLeave={hoverOut} aria-label="Toggle theme">
        <NavIcon name={dark ? "sun" : "moon"} />
      </button>

      <button onClick={onOpenPrefs} className="p-2 rounded-lg border transition-colors" style={{ ...iconBtnStyle, borderColor: "var(--border)" }} onMouseEnter={hoverIn} onMouseLeave={hoverOut} aria-label="Preferences">
        <NavIcon name="settings" />
      </button>
    </header>
  );
}

// ── KPI Cards ───────────────────────────────────────────────────────────────
interface KpiData {
  models: ModelResults;
  backtest: BacktestResults;
  leakage: LeakageAudit;
}

function KpiCards({ kpiData, accent }: { kpiData: KpiData | null; accent: string }) {
  if (!kpiData) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="rounded-xl border p-4 animate-pulse" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
            <div className="h-3 w-24 rounded mb-2" style={{ background: "var(--muted)" }} />
            <div className="h-8 w-16 rounded" style={{ background: "var(--muted)" }} />
          </div>
        ))}
      </div>
    );
  }

  const stackRows = kpiData.models.rows.filter((r) => r.model === "stack_binary");
  const bestAuc   = stackRows.length ? Math.max(...stackRows.map((r) => r.test_auc)) : 0;
  const avgAuc    = stackRows.length ? stackRows.reduce((s, r) => s + r.test_auc, 0) / stackRows.length : 0;
  const bestRow   = stackRows.find((r) => r.test_auc === bestAuc);
  const avgSharpe = kpiData.backtest.summary.length ? kpiData.backtest.summary.reduce((s, r) => s + r.sharpe, 0) / kpiData.backtest.summary.length : 0;
  const avgMdd    = kpiData.backtest.summary.length ? kpiData.backtest.summary.reduce((s, r) => s + r.mdd, 0) / kpiData.backtest.summary.length : 0;
  const leakageOk = kpiData.leakage.modules.every((m) => !m.leakage);

  const kpis = [
    { label: "Best Test AUC",    value: fmt3(bestAuc),            sub: bestRow?.tag ?? "stack",     color: "var(--success)" },
    { label: "Avg Test AUC",     value: fmt3(avgAuc),             sub: "across 4 datasets",         color: accent },
    { label: "Avg Sharpe Ratio", value: avgSharpe.toFixed(2),     sub: "walk-forward 2022–2024",    color: avgSharpe >= 1 ? "var(--success)" : "var(--warn)" },
    { label: "Avg Max Drawdown", value: fmtPct(avgMdd),           sub: "stack equity",              color: "var(--danger)" },
    { label: "Leakage Audit",    value: leakageOk ? "PASS" : "FAIL", sub: `${kpiData.leakage.modules.length} modules`, color: leakageOk ? "var(--success)" : "var(--danger)" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
      {kpis.map((k) => (
        <div
          key={k.label}
          className="rounded-xl border p-4 flex flex-col gap-1 transition-shadow hover:shadow-sm"
          style={{ background: "var(--card, #fff)", borderColor: "var(--border)" }}
          title={GLOSSARY[k.label.toLowerCase().replace("best test ", "").replace("avg test ", "")] ?? undefined}
        >
          <span className="text-[11px] uppercase tracking-wider font-medium" style={{ color: "var(--muted-fg)" }}>
            {k.label}
          </span>
          <div className="text-[26px] font-semibold leading-none tracking-tight tabular-nums" style={{ color: k.color }}>
            {k.value}
          </div>
          <div className="text-[11px] truncate" style={{ color: "var(--muted-fg)" }}>{k.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main Layout ─────────────────────────────────────────────────────────────
export default function DashboardLayout() {
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);
  const [collapsed,  setCollapsed]  = useState(false);
  const [activeTab,  setActiveTab]  = useState("tab1");
  const [prefsOpen,  setPrefsOpen]  = useState(false);
  const [kpiData,    setKpiData]    = useState<KpiData | null>(null);

  // Restore prefs from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("ds108-prefs");
      if (saved) setPrefs({ ...DEFAULT_PREFS, ...JSON.parse(saved) });
    } catch { /* ignore */ }
  }, []);

  // Apply all prefs to DOM
  useEffect(() => {
    const { dark, font, density, accent } = prefs;
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.setAttribute("data-font", font);
    document.documentElement.setAttribute("data-density", density);
    document.documentElement.style.setProperty("--accent-ds", accent);
    try { localStorage.setItem("ds108-prefs", JSON.stringify(prefs)); } catch { /* ignore */ }
  }, [prefs]);

  const setPref = <K extends keyof Prefs>(key: K, val: Prefs[K]) =>
    setPrefs((p) => ({ ...p, [key]: val }));

  // Load KPI data
  useEffect(() => {
    Promise.all([loadModelResults(), loadBacktestResults(), loadLeakageAudit()])
      .then(([models, backtest, leakage]) => setKpiData({ models, backtest, leakage }))
      .catch(() => { /* show skeleton */ });
  }, []);

  const activeNav = NAV.find((n) => n.id === activeTab);

  return (
    <ChartPaletteCtx.Provider value={prefs.palette}>
      <div className="min-h-screen flex" style={{ background: "var(--bg, #fafaf9)" }}>
        <Sidebar
          active={activeTab}
          onSelect={setActiveTab}
          collapsed={collapsed}
          accent={prefs.accent}
        />

        <div className="flex-1 min-w-0 flex flex-col">
          <TopBar
            dark={prefs.dark}
            onToggleDark={() => setPref("dark", !prefs.dark)}
            onToggleSidebar={() => setCollapsed((c) => !c)}
            onOpenPrefs={() => setPrefsOpen((o) => !o)}
          />

          <main className="px-4 lg:px-6 py-6 space-y-6 max-w-[1600px] w-full mx-auto flex-1">
            <KpiCards kpiData={kpiData} accent={prefs.accent} />

            <div className="flex items-baseline justify-between">
              <h2 className="text-[18px] font-semibold tracking-tight" style={{ color: "var(--fg)" }}>
                {activeNav?.label}
              </h2>
              <span className="text-[12px]" style={{ color: "var(--muted-fg)" }}>
                {activeNav?.desc}
              </span>
            </div>

            {activeTab === "tab1" && <Tab1DataQuality />}
            {activeTab === "tab2" && <Tab2Features />}
            {activeTab === "tab3" && <Tab3Leakage />}
            {activeTab === "tab4" && <Tab4Models />}
            {activeTab === "tab5" && <Tab5Hurdle />}

            <footer
              className="pt-6 pb-2 text-[11px] text-center border-t mt-8"
              style={{ color: "var(--muted-fg)", borderColor: "var(--border)" }}
            >
              DS108 Pipeline Dashboard · Coffee &amp; Corn Futures · Walk-forward 2022–2024
            </footer>
          </main>
        </div>

        {prefsOpen && (
          <PreferencesPanel
            prefs={prefs}
            onChange={setPref}
            onClose={() => setPrefsOpen(false)}
          />
        )}
      </div>
    </ChartPaletteCtx.Provider>
  );
}
