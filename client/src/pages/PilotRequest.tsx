import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowLeft, ArrowUpRight, CircleCheck, ShieldCheck } from "lucide-react";
import { Link, useLocation } from "wouter";
import "@/pilot-request.css";
import { pilotRequestSchema, type PilotRequestInput } from "@shared/pilot-schema";
import { apiUrl } from "@/lib/api";

const scopeNotes = [
  {
    label: "No cost",
    detail: "One workload, audited once. Nothing to pay, nothing to sign.",
  },
  {
    label: "No contract",
    detail: "This isn't a service agreement — it's an offer to look at a real run and tell you what I find.",
  },
  {
    label: "Observe first",
    detail: "GhostWatcherHook attaches to your training loop and records telemetry. It does not touch your model or data.",
  },
];

export default function PilotRequest() {
  const [, setLocation] = useLocation();
  const [submitError, setSubmitError] = useState<string | null>(null);

  const form = useForm<PilotRequestInput>({
    resolver: zodResolver(pilotRequestSchema),
    defaultValues: {
      name: "",
      email: "",
      org: "",
      gpuSetup: "",
      workload: "",
      contactPreference: "email",
    },
  });

  async function onSubmit(values: PilotRequestInput) {
    setSubmitError(null);
    try {
      const res = await fetch(apiUrl("/api/pilot-requests"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error === "invalid_request" ? "Check the form for errors." : "Something went wrong submitting this.");
      }
      const { id } = await res.json();
      setLocation(`/pilot/sent?id=${id}`);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Something went wrong submitting this.");
    }
  }

  return (
    <div className="pilot-page">
      <header className="pilot-header">
        <Link href="/" className="pilot-back">
          <ArrowLeft size={14} /> Back
        </Link>
        <span className="orbit-eyebrow"><span /> FREE PILOT REQUEST</span>
      </header>

      <main className="pilot-main">
        <div className="pilot-intro">
          <h1>Start with one real training run.</h1>
          <p>
            GhostLayer is in an early, unpaid pilot phase. Tell me about your setup and
            what you're training, and I'll run an audit against it directly and send you
            what it finds — no cost, no contract, no obligation either way.
          </p>

          <ul className="pilot-scope">
            {scopeNotes.map((note) => (
              <li key={note.label}>
                <ShieldCheck size={15} />
                <div>
                  <b>{note.label}</b>
                  <p>{note.detail}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <form className="pilot-form" onSubmit={form.handleSubmit(onSubmit)} noValidate>
          <div className="pilot-field">
            <label htmlFor="name">Name</label>
            <input id="name" type="text" autoComplete="name" {...form.register("name")} />
            {form.formState.errors.name && (
              <span className="pilot-error">{form.formState.errors.name.message}</span>
            )}
          </div>

          <div className="pilot-field">
            <label htmlFor="email">Email</label>
            <input id="email" type="email" autoComplete="email" {...form.register("email")} />
            {form.formState.errors.email && (
              <span className="pilot-error">{form.formState.errors.email.message}</span>
            )}
          </div>

          <div className="pilot-field">
            <label htmlFor="org">Team or project (optional)</label>
            <input id="org" type="text" {...form.register("org")} />
          </div>

          <div className="pilot-field">
            <label htmlFor="gpuSetup">GPU setup</label>
            <input
              id="gpuSetup"
              type="text"
              placeholder="e.g. 2x RTX 4090, single A100, RunPod A2000"
              {...form.register("gpuSetup")}
            />
            {form.formState.errors.gpuSetup && (
              <span className="pilot-error">{form.formState.errors.gpuSetup.message}</span>
            )}
          </div>

          <div className="pilot-field">
            <label htmlFor="workload">What are you training?</label>
            <textarea
              id="workload"
              rows={5}
              placeholder="Model size, framework (PyTorch / HF Trainer / Lightning / raw loop), what's slow or expensive about it right now."
              {...form.register("workload")}
            />
            {form.formState.errors.workload && (
              <span className="pilot-error">{form.formState.errors.workload.message}</span>
            )}
          </div>

          {submitError && <div className="pilot-error pilot-error--form">{submitError}</div>}

          <button type="submit" className="panel-action panel-action--mail" disabled={form.formState.isSubmitting}>
            {form.formState.isSubmitting ? "Sending…" : "Request a free audit"} <ArrowUpRight size={15} />
          </button>
        </form>
      </main>
    </div>
  );
}

export function PilotRequestSent() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get("id");

  return (
    <div className="pilot-page pilot-page--confirm">
      <main className="pilot-main pilot-main--confirm">
        <CircleCheck size={32} className="pilot-confirm-icon" />
        <h1>Request received.</h1>
        <p>
          I'll follow up by email once I've run the audit against your workload. If you
          want to check back, this is your report link once it's ready:
        </p>
        {id && (
          <Link href={`/report/${id}`} className="pilot-report-link">
            {window.location.origin}/report/{id}
          </Link>
        )}
        <Link href="/" className="panel-action">
          <ArrowLeft size={14} /> Back to GhostLayer
        </Link>
      </main>
    </div>
  );
}
