import type { ReactNode } from "react";

import type { InquiryOut } from "../../api/inquiries";
import { STATE_ICON, STATE_LABEL, URGENCY_ICON, formatDateTime } from "../../lib/ticketDisplay";

interface TicketCardProps {
  inquiry: InquiryOut;
  /** Rendered below the card body -- Phase 3 slots the Workflow/Review/
   * Explanation tabs in here via an expandable wrapper. */
  children?: ReactNode;
}

/** The read-only ticket summary: reference, urgency/SLA line, text, and
 * pills -- mirrors ui/streamlit_app.py's render_tickets() card markup. */
export function TicketCard({ inquiry, children }: TicketCardProps) {
  const state = inquiry.sla_state ?? "on_track";
  const isDuplicate = inquiry.duplicate_of_id != null;
  const classes = ["ticket", state, isDuplicate ? "duplicate" : ""].filter(Boolean).join(" ");

  return (
    <div className={classes}>
      <div className="ticket-top">
        <span>
          {inquiry.reference} · {formatDateTime(inquiry.created_at)}
        </span>
        <span>
          {URGENCY_ICON[inquiry.urgency ?? ""] ?? ""} {inquiry.urgency ?? "—"} priority &nbsp;·&nbsp;{" "}
          {STATE_ICON[state] ?? ""} {STATE_LABEL[state] ?? state} — {inquiry.sla_remaining ?? ""}
        </span>
      </div>
      <div className="ticket-text">{inquiry.raw_text}</div>
      <div>
        <span className="pill">{inquiry.department?.name ?? "Unrouted"}</span>
        <span className="pill">{inquiry.status}</span>
        <span className="pill">{inquiry.assignee?.display_name ?? "Unassigned"}</span>
        {inquiry.routing === "triage" && (
          <span className="pill warn">Low confidence — abstained</span>
        )}
        {inquiry.routing === "out_of_scope" && (
          <span className="pill danger">Possibly out of scope</span>
        )}
        {isDuplicate && (
          <span className="pill info">
            Possible duplicate of inquiry #{inquiry.duplicate_of_id} (
            {Math.round((inquiry.duplicate_score ?? 0) * 100)}% similar)
          </span>
        )}
        {inquiry.escalated_at && <span className="pill danger">Escalated</span>}
        {inquiry.source === "email" && <span className="pill">via email</span>}
      </div>
      {children}
    </div>
  );
}
