import { useQuery } from "@tanstack/react-query";

import { getByDepartment, getOverview } from "../api/analytics";
import { listInquiries } from "../api/inquiries";
import { ExpandableTicket } from "../components/ticket/ExpandableTicket";
import { MetricTile } from "../components/MetricTile";
import { useAuth } from "../auth/useAuth";

export function Overview() {
  const { user } = useAuth();

  const { data: overview } = useQuery({ queryKey: ["analytics", "overview"], queryFn: getOverview });
  const { data: byDepartment } = useQuery({
    queryKey: ["analytics", "departments"],
    queryFn: getByDepartment,
  });
  const { data: urgent } = useQuery({
    queryKey: ["inquiries", "overview-urgent"],
    queryFn: () => listInquiries({ open_only: true, order: "due_asc", limit: 5 }),
    enabled: (overview?.total ?? 0) > 0,
  });

  return (
    <>
      <div className="hero">
        <div className="eyebrow">Operations center</div>
        <h1>Good day, {user?.display_name}</h1>
        <p>A live view of the inquiries you are responsible for.</p>
      </div>

      {!overview ? (
        <p className="caption">Loading…</p>
      ) : overview.total === 0 ? (
        <div className="alert">No inquiries yet. Submit the first one from New inquiry.</div>
      ) : (
        <>
          <div className="metric-row">
            <MetricTile label="Total inquiries" value={overview.total} />
            <MetricTile label="Open queue" value={overview.open} />
            <MetricTile label="Abstained to Triage" value={overview.abstained} />
            <MetricTile
              label="SLA breached"
              value={overview.breached}
              delta={overview.breached ? "needs attention" : undefined}
              warn={overview.breached > 0}
            />
          </div>

          {byDepartment && byDepartment.length > 0 && (
            <>
              <h2 style={{ fontSize: "1.1rem", margin: "1.2rem 0 0.5rem" }}>Department activity</h2>
              <div className="card" style={{ overflowX: "auto" }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Department</th>
                      <th>Inquiries</th>
                      <th>Open</th>
                      <th>Breached</th>
                    </tr>
                  </thead>
                  <tbody>
                    {byDepartment.map((row) => (
                      <tr key={row.code}>
                        <td>{row.name}</td>
                        <td>{row.total}</td>
                        <td>{row.open}</td>
                        <td>{row.breached}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h2 style={{ fontSize: "1.1rem", margin: "1.2rem 0 0.5rem" }}>
            Most urgent open inquiries
          </h2>
          {urgent && urgent.items.length > 0 ? (
            urgent.items.map((inquiry) => <ExpandableTicket key={inquiry.id} inquiry={inquiry} />)
          ) : (
            <div className="alert">
              This queue is clear. New inquiries routed here appear automatically.
            </div>
          )}
        </>
      )}
    </>
  );
}
