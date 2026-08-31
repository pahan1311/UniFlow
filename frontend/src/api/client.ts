/**
 * client.ts
 * ---------
 * The one place that knows how to talk to the backend: attaches the
 * bearer token to every request, retries exactly once after a silent
 * refresh on a 401, and normalizes the API's two error shapes into a
 * single ApiError.
 *
 * Mirrors ui/api_client.py's ApiClient._request() so the browser fails
 * the same way the old Streamlit client did -- one refresh attempt, then
 * a clean sign-out, never a silent retry loop.
 */
import createClient from "openapi-fetch";
import type { paths } from "./schema";
import { tokenStore } from "./token-store";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

// Empty baseUrl: the Vite dev proxy and the production static-file mount
// both make the API same-origin, so every path is relative.
export const api = createClient<paths>({ baseUrl: "" });

api.use({
  onRequest({ request }) {
    const token = tokenStore.getAccess();
    if (token) request.headers.set("Authorization", `Bearer ${token}`);
    return request;
  },
});

let onAuthFailure: (() => void) | null = null;

/** Registered once by AuthContext so client.ts can force a sign-out
 * without importing React state directly. */
export function setAuthFailureHandler(fn: () => void): void {
  onAuthFailure = fn;
}

function detailToMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI's own request-validation shape: [{loc, msg, type}, ...]
    return detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String((item as { msg: unknown }).msg)
          : JSON.stringify(item),
      )
      .join("; ");
  }
  return null;
}

function normalizeError(status: number, body: unknown): ApiError {
  let message = `Request failed (${status})`;
  if (body && typeof body === "object" && "detail" in body) {
    const resolved = detailToMessage((body as { detail: unknown }).detail);
    if (resolved) message = resolved;
  }
  return new ApiError(status, message);
}

async function refreshTokens(): Promise<boolean> {
  const refresh_token = tokenStore.getRefresh();
  if (!refresh_token) return false;

  const { data, error } = await api.POST("/api/v1/auth/refresh", {
    body: { refresh_token },
  });
  if (error || !data) return false;

  tokenStore.set(data.access_token, data.refresh_token, data.user);
  return true;
}

type FetchResult<T> = {
  data?: T;
  error?: unknown;
  response: Response;
};

/**
 * Wraps one openapi-fetch call. `fn` is called again (not memoized) on a
 * 401 after a successful refresh, since the first call's request already
 * carried the now-stale Authorization header.
 */
export async function call<T>(fn: () => Promise<FetchResult<T>>): Promise<T> {
  let result = await fn();

  if (result.response.status === 401) {
    const refreshed = await refreshTokens();
    if (refreshed) {
      result = await fn();
    }
  }

  if (result.response.status === 401) {
    tokenStore.clear();
    onAuthFailure?.();
    throw new ApiError(401, "Your session has ended. Please sign in again.");
  }

  if (!result.response.ok) {
    throw normalizeError(result.response.status, result.error);
  }

  return result.data as T;
}
