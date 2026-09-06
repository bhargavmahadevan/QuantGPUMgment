import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import { pilotRouter } from "./pilotRoutes.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const server = createServer(app);

  app.use(express.json());

  // CORS middleware for split deployment (e.g. Firebase Hosting frontend + Render API backend)
  // Closed by default: if ALLOWED_ORIGIN is unset, no CORS headers are emitted (same-origin only).
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    const allowed = process.env.ALLOWED_ORIGIN;

    if (!allowed || !origin) {
      return next();
    }

    const allowedList = allowed.split(",").map((o) => o.trim());
    const isAllowed = allowed === "*" || allowedList.includes(origin);

    if (isAllowed) {
      res.setHeader("Access-Control-Allow-Origin", allowed === "*" ? "*" : origin);
      if (allowed !== "*") {
        res.setHeader("Vary", "Origin");
      }
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

      if (req.method === "OPTIONS") {
        return res.sendStatus(204);
      }
    }

    next();
  });

  app.use("/api", pilotRouter);

  // Serve static files from dist/public in production
  const staticPath =
    process.env.NODE_ENV === "production"
      ? path.resolve(__dirname, "public")
      : path.resolve(__dirname, "..", "dist", "public");

  app.use(express.static(staticPath));

  // Handle client-side routing - serve index.html for all routes
  app.get("*", (_req, res) => {
    res.sendFile(path.join(staticPath, "index.html"));
  });

  const port = process.env.PORT || 3000;

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
