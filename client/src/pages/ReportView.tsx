import { useEffect, useState } from "react";
import { useParams, Link } from "wouter";
import {
  ArrowLeft,
  Clock3,
  CircleAlert,
  CircleCheck,
  Gauge,
  ShieldCheck,
} from "lucide-react";
import "@/pilot-request.css";
import "@/report-view.css";
import type { PilotFindings, PilotStatus } from "@shared/pilot-schema";
import { apiUrl } from "@/lib/api";

interface ReportData {
  id: string;
  name: string;
  gpuSetup: string;
  workload: string;
  submittedAt: string;
  status: PilotStatus;
  findings?: PilotFindings;
}

const statusCopy: Record<PilotStatus, { label: string; detail: string }> = {
  received: {
    label: "Received",
    detail: "Your pilot request is in the queue. I'll run the audit against your workload and update this page directly — no need to check back constantly.",
  },
  in_progress: {
    label: "Audit in progress",
    detail: "GhostWatcherHook is attached and recording telemetry from your run. This page will update once results are in.",
  },
  delivered: {
    label: "Delivered",
    detail: "Here's what the audit found.",
  },
};

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="report-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function ReportView() {
  const params = useParams<{ id: string }>();
  const [data, setData] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(apiUrl(`/api/pilot-requests/${params.id}`));
        if (res.status === 404) {
          if (!cancelled) setError("No pilot request found for this link.");
          return;
        }
        if (!res.ok) throw new Error("Couldn't load this report right now.");
        const json = (await res.json()) as ReportData;
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Couldn't load this report right now.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  return (
    <div className="pilot-page">
      <header className="pilot-header">
        <Link href="/" className="pilot-back">
          <ArrowLeft size={14} /> Back
        </Link>
        <span className="orbit-eyebrow"><span /> PILOT REPORT</span>
      </header>

      <main className="report-main">
        {loading && <p className="report-loading">Loading…</p>}

        {!loading && error && (
          <div className="report-status report-status--error">
            <CircleAlert size={22} />
            <div>
              <h1>Can't find that report</h1>
              <p>{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && data && (
          <>
            <div className={`report-status report-status--${data.status}`}>
              {data.status === "delivered" ? <CircleCheck size={22} /> : <Clock3 size={22} />}
              <div>
                <h1>{statusCopy[data.status].label}</h1>
                <p>{statusCopy[data.status].detail}</p>
              </div>
            </div>

            <div className="report-context">
              <div>
                <span>Requested</span>
                <strong>{new Date(data.submittedAt).toLocaleDateString()}</strong>
              </div>
              <div>
                <span>GPU setup</span>
                <strong>{data.gpuSetup}</strong>
              </div>
              <div>
                <span>Workload</span>
                <strong>{data.workload}</strong>
              </div>
            </div>

            {data.status === "delivered" && data.findings && (
              <section className="report-findings">
                <h2>Findings</h2>
                <p className="report-findings__summary">{data.findings.summary}</p>

                <div className="report-metrics">
                  {data.findings.hardwareName && (
                    <Metric label="Hardware" value={data.findings.hardwareName} />
                  )}
                  {data.findings.precisionSpeedupPct !== undefined && (
                    <Metric label="Precision speedup" value={`${data.findings.precisionSpeedupPct}%`} />
                  )}
                  {data.findings.workersSpeedupPct !== undefined && (
                    <Metric label="DataLoader worker speedup" value={`${data.findings.workersSpeedupPct}%`} />
                  )}
                  {data.findings.gradCheckpointSavingPct !== undefined && (
                    <Metric label="Grad checkpoint VRAM saving" value={`${data.findings.gradCheckpointSavingPct}%`} />
                  )}
                  {data.findings.recommendedBatchMultiplier !== undefined && (
                    <Metric label="Safe batch multiplier" value={`${data.findings.recommendedBatchMultiplier}x`} />
                  )}
                </div>

                <div className="report-integrity">
                  <ShieldCheck size={14} />
                  {data.findings.isEmpiricallyMeasured
                    ? "Measured directly against your workload, not a published estimate."
                    : "Based on a published reference figure, not measured directly against your workload."}
                </div>

                {data.findings.caveats && (
                  <p className="report-caveats"><Gauge size={13} /> {data.findings.caveats}</p>
                )}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
