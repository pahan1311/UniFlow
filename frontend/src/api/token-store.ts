/**
 * token-store.ts
 * --------------
 * Persists the access/refresh tokens and the signed-in user to
 * localStorage. Kept separate from AuthContext because client.ts needs
 * to read the current access token when building every request's
 * Authorization header, outside of React's render cycle.
 */
import type { components } from "./schema";

export type User = components["schemas"]["UserOut"];

const ACCESS_KEY = "uniflow.access_token";
const REFRESH_KEY = "uniflow.refresh_token";
const USER_KEY = "uniflow.user";

export const tokenStore = {
  getAccess: (): string | null => localStorage.getItem(ACCESS_KEY),
  getRefresh: (): string | null => localStorage.getItem(REFRESH_KEY),
  getUser: (): User | null => {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as User;
    } catch {
      return null;
    }
  },
  set(accessToken: string, refreshToken: string, user: User): void {
    localStorage.setItem(ACCESS_KEY, accessToken);
    localStorage.setItem(REFRESH_KEY, refreshToken);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  clear(): void {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  },
};
