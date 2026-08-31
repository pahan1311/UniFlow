/**
 * AuthContext.tsx
 * ---------------
 * The only piece of client-only state in the app -- everything else is
 * server state managed by React Query. Boots optimistically from
 * whatever is in localStorage (no network round trip just to render the
 * shell); the first real API call's 401-refresh-retry in client.ts is
 * what actually validates the session.
 */
import { createContext, useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import * as authApi from "../api/auth";
import { setAuthFailureHandler } from "../api/client";
import { tokenStore, type User } from "../api/token-store";

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => tokenStore.getUser());

  const clearSession = useCallback(() => {
    tokenStore.clear();
    setUser(null);
  }, []);

  // client.ts calls this when a refresh attempt itself fails (401 twice
  // in a row) -- the session is genuinely over, not just momentarily stale.
  useEffect(() => setAuthFailureHandler(clearSession), [clearSession]);

  const login = useCallback(async (username: string, password: string) => {
    const tokens = await authApi.login(username, password);
    tokenStore.set(tokens.access_token, tokens.refresh_token, tokens.user);
    setUser(tokens.user);
  }, []);

  const logout = useCallback(() => {
    // Best-effort: sign out locally even if the network call fails, the
    // same guarantee ui/api_client.py's logout() makes.
    authApi.logout().catch(() => undefined);
    clearSession();
  }, [clearSession]);

  const value = useMemo<AuthState>(
    () => ({ user, isAuthenticated: user !== null, login, logout }),
    [user, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
