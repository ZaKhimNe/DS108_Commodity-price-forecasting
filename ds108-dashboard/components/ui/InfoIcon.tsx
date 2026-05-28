"use client";

export function InfoIcon({ tip }: { tip: string }) {
  return (
    <span className="relative group inline-flex items-center shrink-0">
      <span
        className="w-4 h-4 rounded-full border text-[10px] font-semibold
                   flex items-center justify-center cursor-help
                   text-[var(--muted-fg)] border-[var(--muted-fg)]"
      >
        ?
      </span>
      <span
        className="pointer-events-none absolute bottom-6 left-0 z-50 w-72
                   rounded-lg border p-3 text-[12px] leading-relaxed
                   shadow-lg opacity-0 group-hover:opacity-100 transition-opacity
                   whitespace-normal"
        style={{
          background: "var(--card, #fff)",
          color: "var(--card-fg, #0a0a0a)",
          borderColor: "var(--border, rgba(0,0,0,.1))",
        }}
      >
        {tip}
      </span>
    </span>
  );
}
