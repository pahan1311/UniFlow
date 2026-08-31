import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { getDepartments, getUsers } from "../../api/admin";
import type { InquiryDetail, QueueStatus } from "../../api/inquiries";
import { routeInquiry, updateInquiry } from "../../api/inquiries";
import { STATE_LABEL, formatDateTime } from "../../lib/ticketDisplay";
import { currentVersion, useVersionedMutation } from "./useVersionedMutation";

const STATUS_OPTIONS: QueueStatus[] = [
  "New",
  "In Progress",
  "Waiting on Student",
  "Resolved",
  "Closed",
];

interface WorkflowTabProps {
  detail: InquiryDetail;
  /** True only in the Triage queue: an abstained inquiry has no owning
   * department yet, so it needs a route-to-department control on top of
   * the usual status/assignee workflow. */
  allowRouting?: boolean;
}

function RouteControl({ detail }: { detail: InquiryDetail }) {
  const queryClient = useQueryClient();
  const { data: departments } = useQuery({
    queryKey: ["departments", "all"],
    queryFn: () => getDepartments(true),
  });
  const realDepartments = (departments ?? []).filter((d) => !d.is_triage);
  const predictedCode = detail.prediction?.category_label;

  const [target, setTarget] = useState<string>(predictedCode ?? "");

  const mutation = useVersionedMutation(detail.id, () => {
    const departmentId = realDepartments.find((d) => d.code === target)?.id;
    if (!departmentId) throw new Error("Choose a department to route to.");
    return routeInquiry(detail.id, {
      department_id: departmentId,
      expected_version: currentVersion(queryClient, detail.id) ?? detail.version,
    });
  });

  return (
    <>
      {mutation.conflict && <div className="alert error">{mutation.conflict}</div>}
      <div style={{ display: "grid", gridTemplateColumns: "3fr 1fr", gap: "0.7rem", alignItems: "end" }}>
        <div className="field-group" style={{ margin: 0 }}>
          <label>Route to department</label>
          <select className="field" value={target} onChange={(e) => setTarget(e.target.value)}>
            {realDepartments.map((d) => (
              <option key={d.code} value={d.code}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="btn primary"
          disabled={mutation.isPending || !target}
          onClick={() => mutation.mutate()}
        >
          {mutation.isPending ? "Routing…" : "Route"}
        </button>
      </div>
      {detail.prediction && (
        <p className="caption" style={{ marginTop: "0.3rem" }}>
          Model suggested {detail.prediction.category_label} at{" "}
          {Math.round(detail.prediction.category_confidence * 100)}% confidence.
        </p>
      )}
      <hr style={{ margin: "1rem 0", border: "none", borderTop: "1px solid var(--line)" }} />
    </>
  );
}

export function WorkflowTab({ detail, allowRouting }: WorkflowTabProps) {
  const queryClient = useQueryClient();
  const { data: users } = useQuery({ queryKey: ["users"], queryFn: getUsers });

  const [status, setStatus] = useState<QueueStatus>(detail.status);
  const [assigneeId, setAssigneeId] = useState<number | "">(detail.assignee?.id ?? "");

  const mutation = useVersionedMutation(detail.id, () =>
    updateInquiry(detail.id, {
      status,
      assignee_id: assigneeId === "" ? null : assigneeId,
      clear_assignee: assigneeId === "",
      expected_version: currentVersion(queryClient, detail.id) ?? detail.version,
    }),
  );

  const confidence = detail.prediction?.category_confidence;

  return (
    <div>
      {allowRouting && <RouteControl detail={detail} />}

      {mutation.conflict && <div className="alert error">{mutation.conflict}</div>}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 2fr 1fr", gap: "0.7rem", alignItems: "end" }}>
        <div className="field-group" style={{ margin: 0 }}>
          <label>Status</label>
          <select
            className="field"
            value={status}
            onChange={(e) => setStatus(e.target.value as QueueStatus)}
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="field-group" style={{ margin: 0 }}>
          <label>Owner</label>
          <select
            className="field"
            value={assigneeId}
            onChange={(e) => setAssigneeId(e.target.value === "" ? "" : Number(e.target.value))}
          >
            <option value="">Unassigned</option>
            {users?.map((u) => (
              <option key={u.id} value={u.id}>
                {u.display_name}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="btn"
          disabled={mutation.isPending}
          onClick={() => mutation.mutate()}
        >
          {mutation.isPending ? "Updating…" : "Update"}
        </button>
      </div>

      <p className="caption" style={{ marginTop: "0.7rem" }}>
        SLA target {formatDateTime(detail.sla_due_at)} ·{" "}
        {STATE_LABEL[detail.sla_state ?? "on_track"]}
        {confidence != null && ` · routing confidence ${Math.round(confidence * 100)}%`}
      </p>
    </div>
  );
}
