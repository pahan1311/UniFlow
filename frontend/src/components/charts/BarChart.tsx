/**
 * BarChart.tsx
 * ------------
 * A single-series vertical bar chart, hand-rolled as inline SVG rather
 * than pulling in a charting library -- the shapes this dashboard needs
 * (a handful of categorical bars) don't justify one.
 */
interface BarChartProps {
  data: Array<{ label: string; value: number }>;
  color?: string;
  height?: number;
  valueFormat?: (value: number) => string;
}

const WIDTH = 640;
const PADDING_LEFT = 40;
const PADDING_BOTTOM = 28;
const PADDING_TOP = 12;

export function BarChart({ data, color = "#00b052", height = 220, valueFormat }: BarChartProps) {
  if (data.length === 0) {
    return <p className="caption">No data yet.</p>;
  }

  const max = Math.max(...data.map((d) => d.value), 1);
  const plotWidth = WIDTH - PADDING_LEFT - 12;
  const plotHeight = height - PADDING_TOP - PADDING_BOTTOM;
  const barGap = 10;
  const barWidth = Math.max(6, plotWidth / data.length - barGap);
  const format = valueFormat ?? ((v: number) => String(v));

  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} width="100%" role="img" aria-label="bar chart">
      {/* gridlines + axis labels at 0%, 50%, 100% of the max value */}
      {[0, 0.5, 1].map((fraction) => {
        const y = PADDING_TOP + plotHeight * (1 - fraction);
        return (
          <g key={fraction}>
            <line
              x1={PADDING_LEFT}
              x2={WIDTH - 12}
              y1={y}
              y2={y}
              stroke="#dcebe3"
              strokeWidth={1}
            />
            <text x={PADDING_LEFT - 8} y={y + 4} textAnchor="end" fontSize={11} fill="#587568">
              {format(Math.round(max * fraction))}
            </text>
          </g>
        );
      })}

      {data.map((d, i) => {
        const barHeight = (d.value / max) * plotHeight;
        const x = PADDING_LEFT + i * (barWidth + barGap) + barGap / 2;
        const y = PADDING_TOP + plotHeight - barHeight;
        return (
          <g key={d.label}>
            <title>
              {d.label}: {format(d.value)}
            </title>
            <rect x={x} y={y} width={barWidth} height={barHeight} rx={4} fill={color} />
            <text
              x={x + barWidth / 2}
              y={height - PADDING_BOTTOM + 16}
              textAnchor="middle"
              fontSize={11}
              fill="#587568"
            >
              {d.label.length > 12 ? `${d.label.slice(0, 11)}…` : d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
