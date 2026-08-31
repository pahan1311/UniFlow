import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { getDepartments } from "../../api/admin";
import type { InquiryDetail, Urgency } from "../../api/inquiries";
import { reviewInquiry } from "../../api/inquiries";
import { currentVersion, useVersionedMutation } from "./useVersionedMutation";

const URGENCY_OPTIONS: NonNullable<Urgency>[] = ["Low", "Medium", "High"];

interface ReviewTabProps {
  detail: InquiryDetail;
}

export function ReviewTab({ detail }: ReviewTabProps) {
  const queryClient = useQueryClient();
  const { data: departments } = useQuery({
    queryKey: ["departments", "all"],
    queryFn: () => getDepartments(true),
  });
  const realDepartments = (departments ?? []).filter((d) => !d.is_triage);

  const predictedCode = detail.prediction?.category_label;
  const predictedUrgency = detail.prediction?.urgency_label as NonNullable<Urgency> | undefined;

  const [correctedCode, setCorrectedCode] = useState<string>(predictedCode ?? "");
  const [correctedUrgency, setCorrectedUrgency] = useState<NonNullable<Urgency>>(
    predictedUrgency ?? "Low",
  );

  const mutation = useVersionedMutation(
    detail.id,
    (action: "accept" | "correct") => {
      const departmentId = realDepartments.find((d) => d.code === correctedCode)?.id;
      return reviewInquiry(detail.id, {
        action,
        ...(action === "correct"
          ? { corrected_department_id: departmentId, corrected_urgency: correctedUrgency }
          : {}),
        expected_version: currentVersion(queryClient, detail.id) ?? detail.version,
      });
    },
  );

  return (
    <div>
      {mutation.conflict && <div className="alert error">{mutation.conflict}</div>}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 2fr 1fr", gap: "0.7rem", alignItems: "end" }}>
        <div className="field-group" style={{ margin: 0 }}>
          <label>Correct department</label>
          <select
            className="field"
            value={correctedCode}
            onChange={(e) => setCorrectedCode(e.target.value)}
          >
            {realDepartments.map((d) => (
              <option key={d.code} value={d.code}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field-group" style={{ margin: 0 }}>
          <label>Correct urgency</label>
          <select
            className="field"
            value={correctedUrgency}
            onChange={(e) => setCorrectedUrgency(e.target.value as NonNullable<Urgency>)}
          >
            {URGENCY_OPTIONS.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          <button
            type="button"
            className="btn"
            title="Prediction was right"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate("accept")}
          >
            ✓
          </button>
          <button
            type="button"
            className="btn primary"
            title="Save this correction"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate("correct")}
          >
            ✎
          </button>
        </div>
      </div>
    </div>
  );
}
