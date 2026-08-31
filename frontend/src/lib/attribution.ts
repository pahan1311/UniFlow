/**
 * attribution.ts
 * --------------
 * Ported from ui/streamlit_app.py's highlight_html(): same thresholds
 * (0.08 / -0.08) and the same color formulas, so a token's shading means
 * the same thing here as it did in the Streamlit dashboard. Green
 * supports the predicted department; amber argues against it.
 */
export function attributionColor(score: number): string {
  if (score > 0.08) {
    return `rgba(22,131,95,${Math.min(Math.abs(score), 1) * 0.55})`;
  }
  if (score < -0.08) {
    return `rgba(214,140,26,${Math.min(Math.abs(score), 1) * 0.45})`;
  }
  return "transparent";
}
