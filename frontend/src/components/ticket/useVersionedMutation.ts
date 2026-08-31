/**
 * useVersionedMutation.ts
 * -----------------------
 * Every inquiry mutation (PATCH, route, review) takes `expected_version`
 * and can come back 409 if someone else changed the ticket first. This
 * hook is the one place that handles that: on success it invalidates the
 * inquiry, the list views, and analytics (since a mutation can move a
 * ticket between departments/states); on a 409 it refetches the inquiry
 * and surfaces a conflict message instead of a generic error.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError } from "../../api/client";

export function useVersionedMutation<TVariables, TResult>(
  inquiryId: number,
  mutationFn: (variables: TVariables) => Promise<TResult>,
) {
  const queryClient = useQueryClient();
  const [conflict, setConflict] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn,
    onSuccess: () => {
      setConflict(null);
      queryClient.invalidateQueries({ queryKey: ["inquiry", inquiryId] });
      queryClient.invalidateQueries({ queryKey: ["inquiries"] });
      queryClient.invalidateQueries({ queryKey: ["analytics"] });
    },
    onError: (error: unknown) => {
      if (error instanceof ApiError && error.status === 409) {
        setConflict(
          "This ticket changed since you loaded it — showing the latest version, please retry.",
        );
        queryClient.invalidateQueries({ queryKey: ["inquiry", inquiryId] });
      }
    },
  });

  return { ...mutation, conflict, clearConflict: () => setConflict(null) };
}

/** Reads the version currently in the cache rather than one captured in a
 * stale render closure, so a background refetch can't make a mutation
 * resend an already-outdated version. */
export function currentVersion(
  queryClient: ReturnType<typeof useQueryClient>,
  inquiryId: number,
): number | undefined {
  return queryClient.getQueryData<{ version: number }>(["inquiry", inquiryId])?.version;
}
