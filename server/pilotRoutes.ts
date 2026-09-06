import { Router } from "express";
import { pilotRequestSchema, pilotUpdateSchema } from "../shared/pilot-schema.js";
import { createPilotRequest, getPilotRequest, updatePilotRequest } from "./pilotStore.js";

export const pilotRouter = Router();

// POST /api/pilot-requests — submit a free pilot audit request.
// No payment, no contract: this just queues a request for the founder
// to run manually against a real training workload.
pilotRouter.post("/pilot-requests", async (req, res) => {
  const parsed = pilotRequestSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "invalid_request", issues: parsed.error.issues });
    return;
  }

  try {
    const record = await createPilotRequest(parsed.data);
    // Report link the requester can bookmark — nothing is behind auth yet,
    // the id itself (12 random chars) is the access control at pilot scale.
    res.status(201).json({ id: record.id, status: record.status });
  } catch (err) {
    console.error("Failed to store pilot request:", err);
    res.status(500).json({ error: "storage_failed" });
  }
});

// GET /api/pilot-requests/:id — status/report lookup for a submitted pilot.
pilotRouter.get("/pilot-requests/:id", async (req, res) => {
  const record = await getPilotRequest(req.params.id);
  if (!record) {
    res.status(404).json({ error: "not_found" });
    return;
  }
  // Don't echo back email/org to anyone poking at the id — only the
  // requester-relevant fields for the report view.
  const { email, ...safe } = record;
  res.json(safe);
});

// PATCH /api/pilot-requests/:id — attach real audit findings once you've
// actually run GhostLayer against the participant's workload, or move the
// status along. Single-operator tool: gated by a shared bearer token set via
// the ADMIN_TOKEN env var. Fails CLOSED (501) if that token isn't configured
// at all, rather than silently accepting unauthenticated writes.
pilotRouter.patch("/pilot-requests/:id", async (req, res) => {
  const adminToken = process.env.ADMIN_TOKEN;
  if (!adminToken) {
    res.status(501).json({ error: "admin_not_configured" });
    return;
  }
  const authHeader = req.header("authorization") ?? "";
  const provided = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  if (provided !== adminToken) {
    res.status(401).json({ error: "unauthorized" });
    return;
  }

  const parsed = pilotUpdateSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "invalid_request", issues: parsed.error.issues });
    return;
  }

  const updated = await updatePilotRequest(req.params.id, parsed.data);
  if (!updated) {
    res.status(404).json({ error: "not_found" });
    return;
  }
  const { email, ...safe } = updated;
  res.json(safe);
});
