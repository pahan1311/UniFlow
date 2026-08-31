interface MetricTileProps {
  label: string;
  value: string | number;
  delta?: string;
  warn?: boolean;
}

export function MetricTile({ label, value, delta, warn }: MetricTileProps) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {delta && <div className={`metric-delta${warn ? " warn" : ""}`}>{delta}</div>}
    </div>
  );
}
