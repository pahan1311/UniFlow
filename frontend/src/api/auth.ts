import { api, call } from "./client";
import type { components } from "./schema";

export type TokenOut = components["schemas"]["TokenOut"];

export function login(username: string, password: string): Promise<TokenOut> {
  return call(() => api.POST("/api/v1/auth/login", { body: { username, password } }));
}

export function logout(everywhere = false): Promise<void> {
  return call(() =>
    api.POST("/api/v1/auth/logout", { params: { query: { everywhere } } }),
  );
}
