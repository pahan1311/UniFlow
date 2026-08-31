import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { getExplanation } from "../../api/inquiries";
import { attributionColor } from "../../lib/attribution";

interface ExplanationTabProps {
  inquiryId: number;
}

/** Integrated Gradients attributions, fetched on demand (not on tab
 * mount) and cached by React Query -- mirrors both the server-side
 * explanations cache and Streamlit's session_state cache key, so
 * re-opening a ticket doesn't recompute the ~32 forward+backward passes. */
export function ExplanationTab({ inquiryId }: ExplanationTabProps) {
  const [requested, setRequested] = useState(false);

  const { data: explanation, isFetching } = useQuery({
    queryKey: ["explanation", inquiryId, "category"],
    queryFn: () => getExplanation(inquiryId, "category"),
    enabled: requested,
    staleTime: Infinity,
  });

  if (!requested) {
    return (
      <div>
        <button type="button" className="btn" onClick={() => setRequested(true)}>
          Explain this routing
        </button>
        <p className="caption" style={{ marginTop: "0.5rem" }}>
          Shows which words drove the department prediction, using Integrated Gradients over
          the classifier itself.
        </p>
      </div>
    );
  }

  if (isFetching || !explanation) {
    return <p className="caption">Attributing the prediction…</p>;
  }

  return (
    <div>
      <p style={{ fontWeight: 700, marginBottom: "0.4rem" }}>Model evidence</p>
      <div style={{ lineHeight: 2, fontSize: "0.95rem" }}>
        {explanation.tokens.map((t, i) => (
          <span
            key={i}
            title={t.attribution.toFixed(3)}
            style={{
              background: attributionColor(t.attribution),
              padding: "0.08rem 0.18rem",
              borderRadius: 4,
            }}
          >
            {t.token}{" "}
          </span>
        ))}
      </div>
      {explanation.keyword_agreement != null && (
        <p className="caption" style={{ marginTop: "0.5rem" }}>
          Agreement with the human urgency-keyword heuristic (Jaccard):{" "}
          {explanation.keyword_agreement.toFixed(2)}
        </p>
      )}
    </div>
  );
}
