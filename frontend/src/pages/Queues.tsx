import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { getDepartments } from "../api/admin";
import { listInquiries } from "../api/inquiries";
import type { ListInquiriesParams } from "../api/inquiries";
import { ExpandableTicket } from "../components/ticket/ExpandableTicket";
import { MetricTile } from "../components/MetricTile";

const FILTERS = ["Open", "Breached", "All", "Resolved"] as const;
type Filter = (typeof FILTERS)[number];

function filterParams(filter: Filter): Partial<ListInquiriesParams> {
  switch (filter) {
    case "Open":
      return { open_only: true };
    case "Resolved":
      return { status: ["Resolved"] };
    case "Breached":
      return { sla_state: ["breached"] };
    default:
      return {};
  }
}

/** Each department tab remembers its own filter for the session, the same
 * way Streamlit's st.segmented_control(key=f"filter_{department}") did. */
export function Queues() {
  const { deptId } = useParams();
  const navigate = useNavigate();
  const [filters, setFilters] = useState<Record<string, Filter>>({});

  const { data: departments } = useQuery({
    queryKey: ["departments", "scoped"],
    queryFn: () => getDepartments(false),
  });
  const workable = (departments ?? []).filter((d) => !d.is_triage);
  const active = workable.find((d) => d.id === Number(deptId)) ?? workable[0];
  const filter: Filter = (active && filters[active.code]) || "Open";

  const { data: inCount } = useQuery({
    queryKey: ["inquiries", "count", active?.id, "open"],
    queryFn: () => listInquiries({ department_id: [active!.id], open_only: true, limit: 1 }),
    enabled: !!active,
  });
  const { data: highCount } = useQuery({
    queryKey: ["inquiries", "count", active?.id, "high"],
    queryFn: () =>
      listInquiries({ department_id: [active!.id], urgency: ["High"], open_only: true, limit: 1 }),
    enabled: !!active,
  });
  const { data: breachedCount } = useQuery({
    queryKey: ["inquiries", "count", active?.id, "breached"],
    queryFn: () => listInquiries({ department_id: [active!.id], sla_state: ["breached"], limit: 1 }),
    enabled: !!active,
  });
  const { data: resolvedCount } = useQuery({
    queryKey: ["inquiries", "count", active?.id, "resolved"],
    queryFn: () => listInquiries({ department_id: [active!.id], status: ["Resolved"], limit: 1 }),
    enabled: !!active,
  });

  const { data: page } = useQuery({
    queryKey: ["inquiries", "queue", active?.id, filter],
    queryFn: () =>
      listInquiries({
        department_id: [active!.id],
        ...filterParams(filter),
        order: "priority",
        limit: 100,
      }),
    enabled: !!active,
  });

  if (!departments) return <p className="caption">Loading…</p>;

  if (workable.length === 0) {
    return (
      <div className="alert warn">
        Your account is not attached to a department. Use the Triage queue, or ask an
        administrator to assign you one.
      </div>
    );
  }

  return (
    <>
      <div className="hero">
        <div className="eyebrow">Team workspaces</div>
        <h1>Department queues</h1>
        <p>Focused workspaces for ownership, prioritisation and resolution.</p>
      </div>

      <div style={{ display: "flex", gap: "0.4rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        {workable.map((d) => (
          <button
            key={d.code}
            type="button"
            className={d.code === active?.code ? "btn selected" : "btn"}
            onClick={() => navigate(`/queues/${d.id}`)}
          >
            {d.name}
          </button>
        ))}
      </div>

      {active && (
        <>
          <div className="metric-row">
            <MetricTile label="In queue" value={inCount?.total ?? "…"} />
            <MetricTile label="High priority" value={highCount?.total ?? "…"} />
            <MetricTile label="Breached" value={breachedCount?.total ?? "…"} />
            <MetricTile label="Resolved" value={resolvedCount?.total ?? "…"} />
          </div>

          <div style={{ display: "flex", gap: "0.4rem", margin: "0.8rem 0" }}>
            {FILTERS.map((f) => (
              <button
                key={f}
                type="button"
                className={f === filter ? "btn selected" : "btn"}
                onClick={() => setFilters((prev) => ({ ...prev, [active.code]: f }))}
              >
                {f}
              </button>
            ))}
          </div>

          {page && page.items.length > 0 ? (
            page.items.map((inquiry) => <ExpandableTicket key={inquiry.id} inquiry={inquiry} />)
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
