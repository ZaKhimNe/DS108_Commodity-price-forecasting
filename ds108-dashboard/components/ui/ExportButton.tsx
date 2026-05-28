"use client";

interface ExportButtonProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rows: Record<string, any>[];
  filename: string;
}

export function ExportButton({ rows, filename }: ExportButtonProps) {
  const download = () => {
    if (!rows.length) return;
    const keys = Object.keys(rows[0]);
    const csv = [
      keys.join(","),
      ...rows.map((r) =>
        keys.map((k) => JSON.stringify(r[k] ?? "")).join(",")
      ),
    ].join("\n");
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })),
      download: filename,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <button
      onClick={download}
      className="text-[11px] px-2.5 py-1 rounded border transition-colors shrink-0"
      style={{
        borderColor: "var(--border, rgba(0,0,0,.1))",
        color: "var(--muted-fg, #737373)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = "var(--fg, #0a0a0a)";
        (e.currentTarget as HTMLButtonElement).style.background = "var(--row-hover, rgba(0,0,0,.025))";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = "var(--muted-fg, #737373)";
        (e.currentTarget as HTMLButtonElement).style.background = "";
      }}
    >
      ↓ CSV
    </button>
  );
}
