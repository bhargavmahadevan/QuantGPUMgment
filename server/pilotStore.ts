import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { customAlphabet } from "nanoid";
import type { PilotRequestInput, PilotRequestRecord, PilotUpdateInput } from "../shared/pilot-schema.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, "..", "data", "pilot_requests");
const DATA_FILE = path.join(DATA_DIR, "requests.json");

// Short, URL-safe, non-guessable ids for /report/:id links.
const genId = customAlphabet("0123456789abcdefghjkmnpqrstvwxyz", 12);

async function ensureStore(): Promise<void> {
  await fs.mkdir(DATA_DIR, { recursive: true });
  try {
    await fs.access(DATA_FILE);
  } catch {
    await fs.writeFile(DATA_FILE, "[]", "utf-8");
  }
}

async function readAll(): Promise<PilotRequestRecord[]> {
  await ensureStore();
  const raw = await fs.readFile(DATA_FILE, "utf-8");
  try {
    return JSON.parse(raw) as PilotRequestRecord[];
  } catch {
    // Corrupt/empty file shouldn't take the whole endpoint down.
    return [];
  }
}

async function writeAll(records: PilotRequestRecord[]): Promise<void> {
  await ensureStore();
  await fs.writeFile(DATA_FILE, JSON.stringify(records, null, 2), "utf-8");
}

export async function createPilotRequest(
  input: PilotRequestInput,
): Promise<PilotRequestRecord> {
  const records = await readAll();
  const record: PilotRequestRecord = {
    ...input,
    id: genId(),
    submittedAt: new Date().toISOString(),
    status: "received",
  };
  records.push(record);
  await writeAll(records);
  return record;
}

export async function getPilotRequest(
  id: string,
): Promise<PilotRequestRecord | undefined> {
  const records = await readAll();
  return records.find((r) => r.id === id);
}

export async function updatePilotRequest(
  id: string,
  patch: PilotUpdateInput,
): Promise<PilotRequestRecord | undefined> {
  const records = await readAll();
  const index = records.findIndex((r) => r.id === id);
  if (index === -1) return undefined;

  records[index] = {
    ...records[index],
    ...(patch.status ? { status: patch.status } : {}),
    ...(patch.findings ? { findings: patch.findings } : {}),
  };
  await writeAll(records);
  return records[index];
}
