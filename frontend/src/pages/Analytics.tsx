import { useQuery } from "@tanstack/react-query";

import { getByDepartment, getOverview, getRoutingMix, getSlaCompliance, getVolume } from "../api/analytics";
import { listInquiries } from "../api/inquiries";
import { BarChart } from "../components/charts/BarChart";
import { StackedBarChart } from "../components/charts/StackedBarChart";
import { MetricTile } from "../components/MetricTile";

const ROUTING_LABELS: Record<string, string> = {
  auto: "Auto-routed",
  triage: "Abstained to Triage",
  out_of_scope: "Flagged out of scope",
};

function toCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return "";
  const headers = Object.keys(rows[0]);
  const escape = (value: unknown) => {
    const s = value == null ? "" : String(value);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => escape(row[h])).join(","));
  }
  return lines.join("\n");
}

function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function Analytics() {
  const { data: overview } = useQuery({ queryKey: ["analytics", "overview"], queryFn: getOverview });
  const { data: sla } = useQuery({ queryKey: ["analytics", "sla"], queryFn: getSlaCompliance });
  const { data: volume } = useQuery({
    queryKey: ["analytics", "volume"],
    queryFn: () => getVolume(30),
  });
  const { data: routingMix } = useQuery({
    queryKey: ["analytics", "routing"],
    queryFn: getRoutingMix,
  });
  const { data: byDepartment } = useQuery({
    queryKey: ["analytics", "departments"],
    queryFn: getByDepartment,
  });

  async function exportCsv() {
    const page = await listInquiries({ limit: 1000 });
    downloadCsv(
      `uniflow_inquiries_${new Date().toISOString().slice(0, 10)}.csv`,
      toCsv(page.items as unknown as Record<string, unknown>[]),
    );
  }

  if (!overview || overview.total === 0) {
    return (
      <>
        <div className="hero">
          <div className="eyebrow">Monitoring</div>
          <h1>Analytics</h1>
          <p>Volume, SLA performance and model health.</p>
        </div>
        <div className="alert">No data yet.</div>
      </>
    );
  }

  const correctionRate = overview.correction_rate;

  return (
    <>
      <div className="hero">
        <div className="eyebrow">Monitoring</div>
        <h1>Analytics</h1>
        <p>Volume, SLA performance and model health.</p>
      </div>

      <div className="metric-row">
        <MetricTile
          label="SLA compliance"
          value={sla?.compliance_rate != null ? `${Math.round(sla.compliance_rate * 100)}%` : "—"}
        />
        <MetricTile label="Breached" value={overview.breached} />
        <MetricTile label="At risk" value={overview.at_risk} />
        <MetricTile
          label="Median resolution"
          value={
            sla?.mean_resolution_hours != null ? `${sla.mean_resolution_hours.toFixed(1)} h` : "—"
          }
        />
      </div>

      <h2 style={{ fontSize: "1.1rem", margin: "1.2rem 0 0.5rem" }}>Inquiry volume over time</h2>
      <div className="card">
        <StackedBarChart
          data={(volume ?? []) as unknown as Record<string, number | string>[]}
          categoryKey="day"
          series={[
            { key: "auto", label: "Auto-routed", color: "#00b052" },
            { key: "triage", label: "Abstained to Triage", color: "#d68c1a" },
            { key: "out_of_scope", label: "Flagged out of scope", color: "#c2410c" },
          ]}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.2rem", marginTop: "1.2rem" }}>
        <div>
          <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Routing decisions</h2>
          <div className="card">
            <BarChart
              data={(routingMix ?? []).map((r) => ({
                label: ROUTING_LABELS[r.routing] ?? r.routing,
                value: r.count,
              }))}
            />
            {overview.coverage != null && (
              <p className="caption" style={{ marginTop: "0.5rem" }}>
                Automation coverage: {Math.round(overview.coverage * 100)}% of inquiries were
                routed without human involvement.
              </p>
            )}
          </div>
        </div>

        <div>
          <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Model health</h2>
          <div className="card">
            {correctionRate == null ? (
              <div className="alert">
                No review events recorded yet. Use the Review tab on any ticket to start
                building the feedback set.
              </div>
            ) : (
              <>
                <MetricTile
                  label="Correction rate (last 50 reviews)"
                  value={`${Math.round(correctionRate * 100)}%`}
                />
                <p className="caption" style={{ marginTop: "0.5rem" }}>
                  A rising correction rate is the earliest available signal of distribution
                  drift: the inquiries arriving no longer look like the ones the model was
                  trained on.
                </p>
                {correctionRate > 0.3 && (
                  <div className="alert warn" style={{ marginTop: "0.5rem" }}>
                    Correction rate above 30%. This is the signal that the deployed model is
                    drifting from what officers are actually seeing.
                  </div>
                )}
              </>
            )}

            <h3 style={{ fontSize: "0.95rem", marginTop: "1rem" }}>
              Mean routing confidence by department
            </h3>
            <BarChart
              data={(byDepartment ?? [])
                .filter((d) => d.mean_confidence != null)
                .map((d) => ({ label: d.name, value: Math.round((d.mean_confidence ?? 0) * 100) }))}
              valueFormat={(v) => `${v}%`}
            />
          </div>
        </div>
      </div>

      <h2 style={{ fontSize: "1.1rem", margin: "1.2rem 0 0.5rem" }}>Export</h2>
      <button type="button" className="btn" onClick={exportCsv}>
        Download all inquiries as CSV
      </button>
    </>
  );
}
