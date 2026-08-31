import { Outlet } from "react-router-dom";

import { Sidebar } from "./Sidebar";

/** The persistent frame around every authenticated page: a fixed sidebar
 * plus whatever page is routed into the content area. */
export function AppLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
