import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { AuthProvider } from "./auth/AuthContext";
import { RequireAdmin, RequireAuth } from "./auth/RequireAuth";
import { Analytics } from "./pages/Analytics";
import { Login } from "./pages/Login";
import { NewInquiry } from "./pages/NewInquiry";
import { Overview } from "./pages/Overview";
import { Queues } from "./pages/Queues";
import { Triage } from "./pages/Triage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: true },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route element={<RequireAuth />}>
              <Route element={<AppLayout />}>
                <Route path="/" element={<Overview />} />
                <Route element={<RequireAdmin />}>
                  <Route path="/inquiries/new" element={<NewInquiry />} />
                </Route>
                <Route path="/queues" element={<Queues />} />
                {/* Numeric id, not the department code: several codes
                    ("Learning Platform / IT Access") contain a literal
                    slash, which breaks a single :param path segment. */}
                <Route path="/queues/:deptId" element={<Queues />} />
                <Route path="/triage" element={<Triage />} />
                <Route path="/analytics" element={<Analytics />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
