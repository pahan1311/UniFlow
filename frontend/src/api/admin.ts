import { api, call } from "./client";
import type { components } from "./schema";

export type Department = components["schemas"]["DepartmentOut"];
export type RoutingPolicy = components["schemas"]["RoutingPolicyOut"];
export type UserOut = components["schemas"]["UserOut"];

export function getDepartments(allDepartments = false): Promise<Department[]> {
  return call(() =>
    api.GET("/api/v1/departments", { params: { query: { all_departments: allDepartments } } }),
  );
}

export function getUsers(): Promise<UserOut[]> {
  return call(() => api.GET("/api/v1/users"));
}

export function getPolicies(): Promise<RoutingPolicy[]> {
  return call(() => api.GET("/api/v1/policies"));
}
