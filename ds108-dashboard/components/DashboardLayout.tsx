"use client";
import { useEffect, useState } from "react";
import { ChartPaletteCtx } from "@/lib/palette";
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

// ── SVG icon set ─────────────────────────────────────────────────────────
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
    default: return null;
  }
}

// ── Nav definition ────────────────────────────────────────────────────────
const NAV = [
  { id: "tab1", label: "Data Quality",  desc: "Splits & base rates",  icon: "database" },
  { id: "tab2", label: "Features",      desc: "Importance & groups",  icon: "list"     },
  { id: "tab3", label: "Leakage Audit", desc: "Modules & ablation",   icon: "shield"   },
  { id: "tab4", label: "Model Results", desc: "Metrics & backtest",   icon: "chart"    },
  { id: "tab5", label: "Hurdle Model",  desc: "Two-stage regression", icon: "split"    },
];

// ── Sidebar ───────────────────────────────────────────────────────────────
function Sidebar({ active, onSelect, collapsed }: {
  active: string;
  onSelect: (id: string) => void;
  collapsed: boolean;
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
      {/* Logo */}
      <div
        className="h-16 px-4 flex items-center gap-2.5 border-b shrink-0"
        style={{ borderColor: "var(--border, rgba(0,0,0,.08))" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-white shrink-0 font-semibold text-[13px] tracking-tight"
          style={{ background: "#3b82f6" }}
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

      {/* Nav items */}
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
              <span className="shrink-0" style={{ color: isActive ? "#3b82f6" : undefined }}>
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

      {/* Footer info */}
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

// ── TopBar ─────────────────────────────────────────────────────────────────
function TopBar({ dark, onToggleDark, onToggleSidebar }: {
  dark: boolean;
  onToggleDark: () => void;
  onToggleSidebar: () => void;
}) {
  return (
    <header
      className="sticky top-0 z-20 h-16 backdrop-blur border-b px-4 flex items-center gap-3 shrink-0"
      style={{
        background: "var(--bg, #fafaf9)",
        borderColor: "var(--border, rgba(0,0,0,.08))",
      }}
    >
      <button
        onClick={onToggleSidebar}
        className="p-2 rounded-lg transition-colors"
        style={{ color: "var(--muted-fg)" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.color = "var(--fg)";
          (e.currentTarget as HTMLElement).style.background = "var(--row-hover)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.color = "var(--muted-fg)";
          (e.currentTarget as HTMLElement).style.background = "";
        }}
        aria-label="Toggle sidebar"
      >
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

      <button
        onClick={onToggleDark}
        className="p-2 rounded-lg border transition-colors"
        style={{ color: "var(--muted-fg)", borderColor: "var(--border, rgba(0,0,0,.08))" }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--fg)"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--muted-fg)"; }}
        aria-label="Toggle theme"
      >
        <NavIcon name={dark ? "sun" : "moon"} />
      </button>
    </header>
  );
}

// ── KPI Cards ──────────────────────────────────────────────────────────────
interface KpiData {
  models: ModelResults;
  backtest: BacktestResults;
  leakage: LeakageAudit;
}

function KpiCards({ kpiData }: { kpiData: KpiData | null }) {
  if (!kpiData) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="rounded-xl border p-4 animate-pulse"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
          >
            <div className="h-3 w-24 rounded mb-2" style={{ background: "var(--muted)" }} />
            <div className="h-8 w-16 rounded" style={{ background: "var(--muted)" }} />
          </div>
        ))}
      </div>
    );
  }

  const stackRows = kpiData.models.rows.filter((r) => r.model === "stack_binary");
  const bestAuc   = stackRows.length ? Math.max(...stackRows.map((r) => r.test_auc)) : 0;
  const avgAuc    = stackRows.length
    ? stackRows.reduce((s, r) => s + r.test_auc, 0) / stackRows.length : 0;
  const bestRow   = stackRows.find((r) => r.test_auc === bestAuc);
  const avgSharpe = kpiData.backtest.summary.length
    ? kpiData.backtest.summary.reduce((s, r) => s + r.sharpe, 0) / kpiData.backtest.summary.length : 0;
  const avgMdd    = kpiData.backtest.summary.length
    ? kpiData.backtest.summary.reduce((s, r) => s + r.mdd, 0) / kpiData.backtest.summary.length : 0;
  const leakageOk = kpiData.leakage.modules.every((m) => !m.leakage);

  const kpis = [
    { label: "Best Test AUC",    value: fmt3(bestAuc),            sub: bestRow?.tag ?? "stack",     color: "var(--success)" },
    { label: "Avg Test AUC",     value: fmt3(avgAuc),             sub: "across 4 datasets",         color: "#3b82f6" },
    { label: "Avg Sharpe Ratio", value: avgSharpe.toFixed(2),     sub: "walk-forward 2022–2024",    color: avgSharpe >= 1 ? "var(--success)" : "var(--warn)" },
    { label: "Avg Max Drawdown", value: fmtPct(avgMdd),           sub: "stack equity",              color: "var(--danger)" },
    { label: "Leakage Audit",    value: leakageOk ? "PASS" : "FAIL", sub: `${kpiData.leakage.modules.length} modules`, color: leakageOk ? "var(--success)" : "var(--danger)" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
      {kpis.map((k) => (
        <div
          key={k.label}
          className="rounded-xl border p-4 flex flex-col gap-1"
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

// ── Main Layout ────────────────────────────────────────────────────────────
export default function DashboardLayout() {
  const [dark, setDark] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState("tab1");
  const [kpiData, setKpiData] = useState<KpiData | null>(null);

  // Restore dark mode from localStorage on mount
  useEffect(() => {
    try {
      if (localStorage.getItem("ds108-dark") === "1") setDark(true);
    } catch { /* ignore */ }
  }, []);

  // Apply dark class + persist
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("ds108-dark", dark ? "1" : "0"); } catch { /* ignore */ }
  }, [dark]);

  // Load KPI data (3 JSON files)
  useEffect(() => {
    Promise.all([loadModelResults(), loadBacktestResults(), loadLeakageAudit()])
      .then(([models, backtest, leakage]) => setKpiData({ models, backtest, leakage }))
      .catch(() => { /* show skeleton instead */ });
  }, []);

  const activeNav = NAV.find((n) => n.id === activeTab);

  return (
    <ChartPaletteCtx.Provider value="default">
      <div className="min-h-screen flex" style={{ background: "var(--bg, #fafaf9)" }}>
        <Sidebar active={activeTab} onSelect={setActiveTab} collapsed={collapsed} />

        <div className="flex-1 min-w-0 flex flex-col">
          <TopBar
            dark={dark}
            onToggleDark={() => setDark((d) => !d)}
            onToggleSidebar={() => setCollapsed((c) => !c)}
          />

          <main className="px-4 lg:px-6 py-6 space-y-6 max-w-[1600px] w-full mx-auto flex-1">
            <KpiCards kpiData={kpiData} />

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
      </div>
    </ChartPaletteCtx.Provider>
  );
}
