import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuth } from "./useAuth";

/** Wraps the authenticated route tree. An unauthenticated visitor is sent
 * to /login, remembering where they were headed so login can return them
 * there afterward. */
export function RequireAuth() {
  const { isAuthenticated } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return <Outlet />;
}

/** Wraps routes an officer should never reach, even by typing the URL
 * directly -- hiding the nav link (AppLayout) is the everyday guard,
 * this is the one that actually enforces it. */
export function RequireAdmin() {
  const { user } = useAuth();

  if (user?.role !== "admin") {
    return <Navigate to="/" replace />;
  }
  return <Outlet />;
}
