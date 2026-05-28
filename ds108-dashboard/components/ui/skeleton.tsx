export function Skeleton({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={`animate-pulse rounded bg-muted ${className ?? ""}`}
      style={style}
    />
  );
}

export function SkeletonTab({
  rows = 5,
  chartHeight = 240,
}: {
  rows?: number;
  chartHeight?: number;
}) {
  return (
    <div className="space-y-5">
      <div className="rounded-xl border p-4 space-y-3">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-3 w-72" />
        {Array.from({ length: rows }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
      {chartHeight > 0 && (
        <div className="rounded-xl border p-4">
          <Skeleton className="h-5 w-32 mb-4" />
          <Skeleton style={{ height: chartHeight }} className="w-full" />
        </div>
      )}
    </div>
  );
}
