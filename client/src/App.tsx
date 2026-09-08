import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import { useEffect } from "react";
import { apiUrl } from "@/lib/api";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import Founder from "./pages/Founder";
import Terms from "./pages/Terms";
import Privacy from "./pages/Privacy";
import Legal from "./pages/Legal";
import PilotRequest, { PilotRequestSent } from "./pages/PilotRequest";
import ReportView from "./pages/ReportView";
import LegalConsentBanner from "./components/LegalConsentBanner";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/founder"} component={Founder} />
      <Route path={"/terms"} component={Terms} />
      <Route path={"/privacy"} component={Privacy} />
      <Route path={"/legal"} component={Legal} />
      <Route path={"/pilot"} component={PilotRequest} />
      <Route path={"/pilot/sent"} component={PilotRequestSent} />
      <Route path={"/report/:id"} component={ReportView} />
      <Route path={"/404"} component={NotFound} />
      {/* Final fallback route */}
      <Route component={NotFound} />
    </Switch>
  );
}

// NOTE: About Theme
// - First choose a default theme according to your design style (dark or light bg), than change color palette in index.css
//   to keep consistent foreground/background color across components
// - If you want to make theme switchable, pass `switchable` ThemeProvider and use `useTheme` hook

function App() {
  useEffect(() => {
    // Silently pre-warm the Render backend in the background so it wakes up while visitors browse
    fetch(apiUrl("/api/health")).catch(() => {});
  }, []);

  return (
    <ErrorBoundary>
      <ThemeProvider
        defaultTheme="light"
        // switchable
      >
        <TooltipProvider>
          <Toaster />
          <Router />
          <LegalConsentBanner />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
