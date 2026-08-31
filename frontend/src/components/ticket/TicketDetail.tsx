import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { getInquiry } from "../../api/inquiries";
import { ExplanationTab } from "./ExplanationTab";
import { ReviewTab } from "./ReviewTab";
import { WorkflowTab } from "./WorkflowTab";

const TABS = ["Workflow", "Review", "Why this route?"] as const;
type Tab = (typeof TABS)[number];

interface TicketDetailProps {
  inquiryId: number;
  allowRouting?: boolean;
}

/** Fetches the full InquiryDetail (list rows only carry the InquiryOut
 * subset) and renders the three-tab manage pattern shared across Overview,
 * Department queues and Triage. */
export function TicketDetail({ inquiryId, allowRouting }: TicketDetailProps) {
  const [tab, setTab] = useState<Tab>("Workflow");

  const { data: detail, isLoading } = useQuery({
    queryKey: ["inquiry", inquiryId],
    queryFn: () => getInquiry(inquiryId),
  });

  if (isLoading || !detail) {
    return <p className="caption">Loading…</p>;
  }

  return (
    <div style={{ marginTop: "0.7rem", borderTop: "1px solid var(--line)", paddingTop: "0.8rem" }}>
      <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.8rem" }}>
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            className={tab === t ? "btn selected" : "btn"}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Workflow" && <WorkflowTab detail={detail} allowRouting={allowRouting} />}
      {tab === "Review" && <ReviewTab detail={detail} />}
      {tab === "Why this route?" && <ExplanationTab inquiryId={detail.id} />}
    </div>
  );
}
