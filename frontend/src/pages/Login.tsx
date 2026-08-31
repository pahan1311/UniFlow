import { useState } from "react";
import type { FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../auth/useAuth";

const DEMO_ACCOUNTS: Array<[string, string, string]> = [
  ["admin", "admin123", "every department"],
  ["exams", "exams123", "Examinations + Triage"],
  ["finance", "finance123", "Fees and Payments + Triage"],
  ["it", "it123", "Learning Platform / IT Access + Triage"],
  ["registry", "registry123", "Academic Documents + Triage"],
  ["admissions", "admissions123", "Admissions + Triage"],
  ["timetable", "timetable123", "Timetable and Scheduling + Triage"],
  ["general", "general123", "General Administration + Triage"],
];

export function Login() {
  const { isAuthenticated, login } = useAuth();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showDemo, setShowDemo] = useState(false);

  if (isAuthenticated) {
    const from = (location.state as { from?: Location } | null)?.from;
    return <Navigate to={from?.pathname ?? "/"} replace />;
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign-in failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="hero">
          <div className="eyebrow">UniFlow</div>
          <h1>University Inquiry Operations</h1>
          <p>Sign in to reach your department queue.</p>
        </div>

        <form onSubmit={handleSubmit} className="card">
          <div className="field-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              className="field"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
              autoComplete="username"
            />
          </div>
          <div className="field-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              className="field"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>
          {error && <div className="alert error">{error}</div>}
          <button
            type="submit"
            className="btn primary"
            style={{ width: "100%" }}
            disabled={submitting || !username || !password}
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <button
          type="button"
          className="btn"
          style={{ marginTop: "0.8rem", width: "100%" }}
          onClick={() => setShowDemo((v) => !v)}
        >
          {showDemo ? "Hide demo accounts" : "Show demo accounts"}
        </button>
        {showDemo && (
          <table className="data-table" style={{ marginTop: "0.8rem" }}>
            <thead>
              <tr>
                <th>Username</th>
                <th>Password</th>
                <th>Sees</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_ACCOUNTS.map(([user, pass, scope]) => (
                <tr key={user}>
                  <td>
                    <code>{user}</code>
                  </td>
                  <td>
                    <code>{pass}</code>
                  </td>
                  <td>{scope}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
