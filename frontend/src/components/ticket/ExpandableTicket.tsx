import { useState } from "react";

import type { InquiryOut } from "../../api/inquiries";
import { TicketCard } from "./TicketCard";
import { TicketDetail } from "./TicketDetail";

interface ExpandableTicketProps {
  inquiry: InquiryOut;
  /** True only in the Triage queue, where an abstained inquiry has no
   * owning department yet and needs a route-to-department control. */
  allowRouting?: boolean;
}

/** A ticket card with a "Manage" toggle that reveals the three-tab detail
 * view -- mirrors ui/streamlit_app.py's st.expander("Manage ..."). The
 * detail is only fetched once expanded, not for every card in a list. */
export function ExpandableTicket({ inquiry, allowRouting }: ExpandableTicketProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <TicketCard inquiry={inquiry}>
      <button
        type="button"
        className="btn"
        style={{ marginTop: "0.5rem" }}
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? "Hide" : "Manage"} {inquiry.reference}
      </button>
      {expanded && <TicketDetail inquiryId={inquiry.id} allowRouting={allowRouting} />}
    </TicketCard>
  );
}
