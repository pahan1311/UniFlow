import { api, call } from "./client";

export function getReady() {
  return call(() => api.GET("/readyz"));
}
