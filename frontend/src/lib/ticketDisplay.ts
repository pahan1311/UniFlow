/** Shared display maps, ported from ui/streamlit_app.py so both apps read tickets identically. */

export const URGENCY_ICON: Record<string, string> = { High: "🔴", Medium: "🟠", Low: "🟢" };

export const STATE_ICON: Record<string, string> = {
  met: "✅",
  on_track: "🟢",
  at_risk: "🟡",
  breached: "⛔",
};

export const STATE_LABEL: Record<string, string> = {
  met: "Met",
  on_track: "On track",
  at_risk: "At risk",
  breached: "Breached",
};

export const URGENCY_RANK: Record<string, number> = { High: 0, Medium: 1, Low: 2 };

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19);
}
