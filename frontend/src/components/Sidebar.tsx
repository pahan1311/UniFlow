import { useQuery } from "@tanstack/react-query";
import { NavLink } from "react-router-dom";

import { getPolicies } from "../api/admin";
import { getReady } from "../api/health";
import { useAuth } from "../auth/useAuth";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/inquiries/new", label: "New inquiry", adminOnly: true },
  { to: "/queues", label: "Department queues" },
  { to: "/triage", label: "Triage" },
  { to: "/analytics", label: "Analytics" },
];

/** Branding, nav, signed-in user, backend/model health, the active
 * routing policy, and sign-out -- mirrors ui/streamlit_app.py's sidebar.
 * A standalone component (not inlined in AppLayout) and pinned to the
 * viewport in theme.css's .sidebar rule, so its height never stretches
 * to match however tall the main content happens to be. */
export function Sidebar() {
  const { user, logout } = useAuth();

  const { data: ready } = useQuery({
    queryKey: ["readyz"],
    queryFn: getReady,
    refetchInterval: 30_000,
  });
  const { data: policies } = useQuery({
    queryKey: ["policies"],
    queryFn: getPolicies,
  });
  const activePolicy = policies?.find((p) => p.is_active);
  const calibrated = Boolean(
    activePolicy?.derived_from && (activePolicy.derived_from as { calibrated?: boolean }).calibrated,
  );

  return (
    <aside className="sidebar">
      <h2>◈ UniFlow</h2>
      <div className="caption">INQUIRY OPERATIONS</div>
      <hr />
      <div style={{ fontWeight: 700 }}>{user?.display_name}</div>
      <div className="caption">
        {user?.role === "admin" ? "All departments" : user?.department?.name ?? "No department"}
      </div>
      <hr />
      <nav>
        {NAV_ITEMS.filter((item) => !item.adminOnly || user?.role === "admin").map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => (isActive ? "active" : undefined)}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div style={{ flex: 1 }} />
      <hr />
      <div className="caption" style={{ marginBottom: "0.3rem" }}>
        AI ROUTING ENGINE
      </div>
      {ready?.status === "ready" ? (
        <div style={{ color: "#8fe3bd", fontSize: "0.85rem" }}>
          ● Online — {ready.models.backend as string}
        </div>
      ) : (
        <div style={{ color: "#f3a08a", fontSize: "0.85rem" }}>● Backend not ready</div>
      )}
      {activePolicy && (
        <>
          <div className="caption">
            Auto-route above {(activePolicy.category_tau * 100).toFixed(0)}% confidence
          </div>
          <div className="caption">
            {calibrated
              ? `Calibrated (T=${activePolicy.category_temperature.toFixed(2)})`
              : "Uncalibrated policy in force"}
          </div>
        </>
      )}
      <hr />
      <button type="button" className="btn" style={{ width: "100%" }} onClick={logout}>
        Sign out
      </button>
    </aside>
  );
}
