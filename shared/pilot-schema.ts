import { z } from "zod";

/**
 * A GhostLayer pilot request. This is intentionally NOT a purchase, invoice,
 * or service agreement — see technical_docs/01_revenue_and_commercial for why
 * the founder is running a no-cost, no-contract pilot phase for now.
 */
export const pilotRequestSchema = z.object({
  name: z.string().trim().min(1, "Enter your name").max(120),
  email: z.string().trim().email("Enter a valid email"),
  org: z.string().trim().max(160).optional().or(z.literal("")),
  gpuSetup: z
    .string()
    .trim()
    .min(1, "Describe your GPU setup")
    .max(300),
  workload: z
    .string()
    .trim()
    .min(1, "Describe what you're training")
    .max(1000),
  contactPreference: z.enum(["email", "no-preference"]),
});

export type PilotRequestInput = z.infer<typeof pilotRequestSchema>;

/**
 * The result of an actual audit run against a pilot participant's workload.
 * This is a summary shape you fill in by hand after running GhostLayer's
 * real Python pipeline (CalibrationRunner, decision engine, etc.) against
 * their training loop — it is not generated automatically by this web app,
 * which has no wiring into the Python side yet. See the metric fields below
 * as a checklist of what an honest report should include, mirroring the
 * fields the audit-engine code actually measures.
 */
export const pilotFindingsSchema = z.object({
  summary: z.string().trim().min(1).max(4000),
  hardwareName: z.string().trim().max(120).optional(),
  precisionSpeedupPct: z.number().min(0).max(100).optional(),
  workersSpeedupPct: z.number().min(0).max(100).optional(),
  gradCheckpointSavingPct: z.number().min(0).max(100).optional(),
  recommendedBatchMultiplier: z.number().min(1).max(10).optional(),
  isEmpiricallyMeasured: z.boolean().default(true),
  caveats: z.string().trim().max(2000).optional(),
});

export type PilotFindings = z.infer<typeof pilotFindingsSchema>;

export const pilotStatusSchema = z.enum(["received", "in_progress", "delivered"]);
export type PilotStatus = z.infer<typeof pilotStatusSchema>;

export const pilotUpdateSchema = z.object({
  status: pilotStatusSchema.optional(),
  findings: pilotFindingsSchema.optional(),
});
export type PilotUpdateInput = z.infer<typeof pilotUpdateSchema>;

export interface PilotRequestRecord extends PilotRequestInput {
  id: string;
  submittedAt: string;
  status: PilotStatus;
  findings?: PilotFindings;
}
