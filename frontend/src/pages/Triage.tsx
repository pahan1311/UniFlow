import { useQuery } from "@tanstack/react-query";

import { getDepartments, getPolicies } from "../api/admin";
import { listInquiries } from "../api/inquiries";
import { ExpandableTicket } from "../components/ticket/ExpandableTicket";
import { MetricTile } from "../components/MetricTile";

export function Triage() {
  const { data: departments } = useQuery({
    queryKey: ["departments", "scoped"],
    queryFn: () => getDepartments(false),
  });
  const triage = departments?.find((d) => d.is_triage);

  const { data: policies } = useQuery({ queryKey: ["policies"], queryFn: getPolicies });
  const activePolicy = policies?.find((p) => p.is_active);

  const { data: awaiting } = useQuery({
    queryKey: ["inquiries", "count", triage?.id, "open"],
    queryFn: () => listInquiries({ department_id: [triage!.id], open_only: true, limit: 1 }),
    enabled: !!triage,
  });
  const { data: lowConfidence } = useQuery({
    queryKey: ["inquiries", "count", triage?.id, "triage"],
    queryFn: () => listInquiries({ department_id: [triage!.id], routing: ["triage"], limit: 1 }),
    enabled: !!triage,
  });
  const { data: outOfScope } = useQuery({
    queryKey: ["inquiries", "count", triage?.id, "out_of_scope"],
    queryFn: () =>
      listInquiries({ department_id: [triage!.id], routing: ["out_of_scope"], limit: 1 }),
    enabled: !!triage,
  });

  const { data: page } = useQuery({
    queryKey: ["inquiries", "triage-list", triage?.id],
    queryFn: () =>
      listInquiries({ department_id: [triage!.id], open_only: true, order: "priority", limit: 100 }),
    enabled: !!triage,
  });

  if (!departments) return <p className="caption">Loading…</p>;

  if (!triage) {
    return <div className="alert error">No Triage department is configured. Run the seed script.</div>;
  }

  return (
    <>
      <div className="hero">
        <div className="eyebrow">Selective prediction</div>
        <h1>Triage queue</h1>
        <p>Inquiries the model declined to route automatically. A human decides.</p>
      </div>

      <div className="metric-row">
        <MetricTile label="Awaiting placement" value={awaiting?.total ?? "…"} />
        <MetricTile label="Low confidence" value={lowConfidence?.total ?? "…"} />
        <MetricTile label="Possibly out of scope" value={outOfScope?.total ?? "…"} />
      </div>

      {activePolicy && (
        <p className="caption" style={{ margin: "0.5rem 0 1rem" }}>
          The model abstains below a calibrated confidence of{" "}
          {(activePolicy.category_tau * 100).toFixed(0)}%. That threshold is chosen from the
          risk–coverage analysis in evaluation/selective.py, trading automation coverage against
          misrouting risk.
        </p>
      )}

      {page && page.items.length > 0 ? (
        page.items.map((inquiry) => (
          <ExpandableTicket key={inquiry.id} inquiry={inquiry} allowRouting />
        ))
      ) : (
        <div className="alert">
          This queue is clear. New inquiries routed here appear automatically.
        </div>
      )}
    </>
  );
}
