import { useContext } from "react";

import { AuthContext, type AuthState } from "./AuthContext";

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
}
