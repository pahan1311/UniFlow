/**
 * StackedBarChart.tsx
 * -------------------
 * Multi-series stacked bars for the volume-over-time view -- inline SVG,
 * same reasoning as BarChart.tsx (no charting library for a handful of
 * categorical bars).
 */
interface Series {
  key: string;
  label: string;
  color: string;
}

interface StackedBarChartProps {
  data: Array<Record<string, number | string>>;
  categoryKey: string;
  series: Series[];
  height?: number;
}

const WIDTH = 640;
const PADDING_LEFT = 40;
const PADDING_BOTTOM = 36;
const PADDING_TOP = 12;

export function StackedBarChart({ data, categoryKey, series, height = 240 }: StackedBarChartProps) {
  if (data.length === 0) {
    return <p className="caption">No data yet.</p>;
  }

  const totals = data.map((row) => series.reduce((sum, s) => sum + (Number(row[s.key]) || 0), 0));
  const max = Math.max(...totals, 1);
  const plotWidth = WIDTH - PADDING_LEFT - 12;
  const plotHeight = height - PADDING_TOP - PADDING_BOTTOM;
  const barGap = 6;
  const barWidth = Math.max(4, plotWidth / data.length - barGap);
  // Thin out x-axis labels so they don't overlap when there are many days.
  const labelEvery = Math.max(1, Math.ceil(data.length / 10));

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${height}`} width="100%" role="img" aria-label="stacked bar chart">
        {[0, 0.5, 1].map((fraction) => {
          const y = PADDING_TOP + plotHeight * (1 - fraction);
          return (
            <line
              key={fraction}
              x1={PADDING_LEFT}
              x2={WIDTH - 12}
              y1={y}
              y2={y}
              stroke="#dcebe3"
              strokeWidth={1}
            />
          );
        })}

        {data.map((row, i) => {
          const x = PADDING_LEFT + i * (barWidth + barGap) + barGap / 2;
          let cursor = PADDING_TOP + plotHeight;
          return (
            <g key={String(row[categoryKey])}>
              <title>
                {String(row[categoryKey])}:{" "}
                {series.map((s) => `${s.label} ${row[s.key] ?? 0}`).join(", ")}
              </title>
              {series.map((s) => {
                const value = Number(row[s.key]) || 0;
                const segmentHeight = (value / max) * plotHeight;
                cursor -= segmentHeight;
                return (
                  <rect
                    key={s.key}
                    x={x}
                    y={cursor}
                    width={barWidth}
                    height={segmentHeight}
                    fill={s.color}
                  />
                );
              })}
              {i % labelEvery === 0 && (
                <text
                  x={x + barWidth / 2}
                  y={height - PADDING_BOTTOM + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill="#587568"
                >
                  {String(row[categoryKey]).slice(5)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: "1rem", fontSize: "0.8rem", marginTop: "0.3rem" }}>
        {series.map((s) => (
          <span key={s.key} style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 3,
                background: s.color,
                display: "inline-block",
              }}
            />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
