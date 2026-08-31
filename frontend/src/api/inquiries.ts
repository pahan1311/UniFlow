import { api, call } from "./client";
import type { components, paths } from "./schema";

export type InquiryOut = components["schemas"]["InquiryOut"];
export type InquiryDetail = components["schemas"]["InquiryDetail"];
export type InquiryPage = components["schemas"]["InquiryPage"];
export type IntakeResponse = components["schemas"]["IntakeResponse"];
export type ReviewOut = components["schemas"]["ReviewOut"];
export type EventOut = components["schemas"]["EventOut"];
export type ExplanationOut = components["schemas"]["ExplanationOut"];
export type DuplicateOut = components["schemas"]["DuplicateOut"];
export type QueueStatus = components["schemas"]["QueueStatus"];
export type Urgency = components["schemas"]["Urgency"];
export type RoutingDecision = components["schemas"]["RoutingDecision"];
export type ReviewAction = components["schemas"]["ReviewRequest"]["action"];

// Reused directly from the generated schema rather than hand-typed, so a
// new filter param added to the backend shows up here automatically.
export type ListInquiriesParams = NonNullable<
  paths["/api/v1/inquiries"]["get"]["parameters"]["query"]
>;

export function listInquiries(params: ListInquiriesParams = {}): Promise<InquiryPage> {
  return call(() => api.GET("/api/v1/inquiries", { params: { query: params } }));
}

export function getInquiry(id: number): Promise<InquiryDetail> {
  return call(() => api.GET("/api/v1/inquiries/{inquiry_id}", { params: { path: { inquiry_id: id } } }));
}

export function createInquiry(payload: {
  text: string;
  student_email?: string | null;
  student_ref?: string | null;
  source?: "web" | "email" | "api" | "import";
}): Promise<IntakeResponse> {
  return call(() => api.POST("/api/v1/inquiries", { body: { source: "web", ...payload } }));
}

export function updateInquiry(
  id: number,
  payload: {
    status?: string | null;
    assignee_id?: number | null;
    clear_assignee?: boolean;
    expected_version?: number | null;
  },
): Promise<InquiryDetail> {
  return call(() =>
    api.PATCH("/api/v1/inquiries/{inquiry_id}", {
      params: { path: { inquiry_id: id } },
      // Status is a plain string in ListInquiriesParams-adjacent call sites
      // but the generated type wants the literal QueueStatus union --
      // callers pass a real status string that already satisfies it.
      body: payload as components["schemas"]["StatusUpdate"],
    }),
  );
}

export function routeInquiry(
  id: number,
  payload: { department_id: number; note?: string | null; expected_version?: number | null },
): Promise<InquiryDetail> {
  return call(() =>
    api.POST("/api/v1/inquiries/{inquiry_id}/route", {
      params: { path: { inquiry_id: id } },
      body: payload,
    }),
  );
}

export function reviewInquiry(
  id: number,
  payload: {
    action: "accept" | "correct";
    corrected_department_id?: number | null;
    corrected_urgency?: "Low" | "Medium" | "High" | null;
    note?: string | null;
    expected_version?: number | null;
  },
): Promise<ReviewOut> {
  return call(() =>
    api.POST("/api/v1/inquiries/{inquiry_id}/review", {
      params: { path: { inquiry_id: id } },
      body: payload,
    }),
  );
}

export function getExplanation(
  id: number,
  task: "category" | "urgency" = "category",
): Promise<ExplanationOut> {
  return call(() =>
    api.GET("/api/v1/inquiries/{inquiry_id}/explanation", {
      params: { path: { inquiry_id: id }, query: { task } },
    }),
  );
}

export function getEvents(id: number): Promise<EventOut[]> {
  return call(() =>
    api.GET("/api/v1/inquiries/{inquiry_id}/events", { params: { path: { inquiry_id: id } } }),
  );
}
