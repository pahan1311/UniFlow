import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { createInquiry } from "../api/inquiries";

export function NewInquiry() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => createInquiry({ text }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["inquiries"] });
      queryClient.invalidateQueries({ queryKey: ["analytics"] });
      // The ticket appearing in the destination queue is the
      // confirmation -- no separate inline result screen to click past.
      if (data.inquiry.routing === "auto" && data.inquiry.department) {
        navigate(`/queues/${data.inquiry.department.id}`);
      } else {
        navigate("/triage");
      }
    },
    onError: (err) => {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
    },
  });

  function handleSubmit() {
    if (!text.trim()) return;
    mutation.mutate();
  }

  return (
    <>
      <div className="hero">
        <div className="eyebrow">Smart intake</div>
        <h1>Route a new inquiry</h1>
        <p>The model classifies department and urgency, then routes — or abstains.</p>
      </div>

      <div className="card">
        <p style={{ fontWeight: 700, marginBottom: "0.4rem" }}>Student message</p>
        <textarea
          className="field"
          rows={7}
          placeholder="Describe the issue, relevant dates, and what help is needed…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <p className="caption" style={{ margin: "0.5rem 0" }}>
          Urgency is predicted directly from the text — no deadline field is required. When
          confidence falls below the calibrated threshold the system abstains and holds the
          inquiry in Triage rather than guessing. On submission you'll be taken straight to
          wherever it lands.
        </p>
        {error && <div className="alert error">{error}</div>}
        <button
          type="button"
          className="btn primary"
          style={{ width: "100%" }}
          disabled={mutation.isPending || !text.trim()}
          onClick={handleSubmit}
        >
          {mutation.isPending ? "Classifying and routing…" : "Classify & route inquiry  →"}
        </button>
      </div>
    </>
  );
}
