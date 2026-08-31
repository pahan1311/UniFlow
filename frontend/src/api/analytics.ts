/**
 * analytics.ts
 * ------------
 * The analytics endpoints return plain `dict`/`list[dict]` in Python
 * (uniflow/domain/analytics.py), so FastAPI's OpenAPI schema can't infer
 * field-level types for them -- openapi-typescript generates
 * `{[key: string]: unknown}`. These interfaces are hand-written to match
 * exactly what that module actually returns; if its shape changes, update
 * here too.
 */
import { api, call } from "./client";

export interface Overview {
  total: number;
  open: number;
  abstained: number;
  escalated: number;
  duplicates: number;
  breached: number;
  at_risk: number;
  coverage: number | null;
  correction_rate: number | null;
}

export interface DepartmentStats {
  code: string;
  name: string;
  total: number;
  open: number;
  breached: number;
  mean_confidence: number | null;
}

export interface SlaCompliance {
  closed: number;
  met: number;
  compliance_rate: number | null;
  mean_resolution_hours: number | null;
  open_by_state: Record<string, number>;
}

export interface VolumePoint {
  day: string;
  total: number;
  auto: number;
  triage: number;
  out_of_scope: number;
}

export interface RoutingCount {
  routing: string;
  count: number;
}

export interface ConfidenceBucket {
  lower: number;
  upper: number;
  abstained: boolean;
  count: number;
}

export interface ReviewActivityPoint {
  day: string;
  action: string;
  count: number;
}

export function getOverview(): Promise<Overview> {
  return call(() => api.GET("/api/v1/analytics/overview")) as unknown as Promise<Overview>;
}

export function getByDepartment(): Promise<DepartmentStats[]> {
  return call(() => api.GET("/api/v1/analytics/departments")) as unknown as Promise<DepartmentStats[]>;
}

export function getSlaCompliance(): Promise<SlaCompliance> {
  return call(() => api.GET("/api/v1/analytics/sla")) as unknown as Promise<SlaCompliance>;
}

export function getVolume(days = 30): Promise<VolumePoint[]> {
  return call(() =>
    api.GET("/api/v1/analytics/volume", { params: { query: { days } } }),
  ) as unknown as Promise<VolumePoint[]>;
}

export function getRoutingMix(): Promise<RoutingCount[]> {
  return call(() => api.GET("/api/v1/analytics/routing")) as unknown as Promise<RoutingCount[]>;
}

export function getConfidenceHistogram(buckets = 10): Promise<ConfidenceBucket[]> {
  return call(() =>
    api.GET("/api/v1/analytics/confidence", { params: { query: { buckets } } }),
  ) as unknown as Promise<ConfidenceBucket[]>;
}

export function getReviewActivity(days = 30): Promise<ReviewActivityPoint[]> {
  return call(() =>
    api.GET("/api/v1/analytics/reviews", { params: { query: { days } } }),
  ) as unknown as Promise<ReviewActivityPoint[]>;
}
