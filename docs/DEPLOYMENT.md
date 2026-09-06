# GhostLayer Pilot Site — Deployment Runbook & Checklist

This document details the exact operational deployment runbook for hosting the **GhostLayer** pilot site across **Render** (Express API backend) and **Firebase Hosting** (Vite/React frontend).

---

## Architecture Overview

```
[Browser / Pilot Visitor]
       │
       ├──► Frontend: Firebase Hosting (https://ghostlayer-ai.web.app)
       │       └── Communicates via client/src/lib/api.ts (apiUrl)
       │
       └──► Backend: Render Web Service (https://<YOUR_API_URL>.onrender.com)
               └── Express API with Closed-by-Default CORS
```

---

## 0. One-Time Setup: Admin Token

Generate a cryptographically secure 24-byte hex token to protect the `PATCH /api/pilot-requests/:id` findings endpoint:

```bash
# Python
python -c "import secrets; print(secrets.token_hex(24))"

# or OpenSSL
openssl rand -hex 24
```

Store this token securely in your password manager. You will set this as `ADMIN_TOKEN` in Render.

---

## 1. Push to GitHub

Commit the verified deployment-ready files and push to `main`:

```bash
git add -A
git commit -m "feat(pilot): deployment-ready with closed-by-default CORS"
git push origin main
```

---

## 2. Deploy Backend (Render Web Service)

1. Navigate to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service** → Connect your repository.
2. Configure settings:
   - **Environment:** Node
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `node dist/index.js`
3. Set **Environment Variables**:
   | Key | Value | Purpose |
   | :--- | :--- | :--- |
   | `NODE_ENV` | `production` | Production optimizations |
   | `ALLOWED_ORIGIN` | `https://ghostlayer-ai.web.app` | Whitelists the production frontend |
   | `ADMIN_TOKEN` | `<YOUR_ADMIN_TOKEN>` | Secures the report update endpoint |

   *(Optional)* If you maintain a staging Firebase domain, provide both origins comma-separated:
   `https://ghostlayer-ai.web.app, https://staging.ghostlayer-ai.web.app`

4. Click **Deploy Web Service**.
5. Once deployed, note your service URL (e.g. `https://ghostlayer-api.onrender.com`).

---

## 3. Build & Deploy Frontend (Firebase Hosting)

In your local terminal, build the client bundle pointing at the live backend URL:

### PowerShell (Windows):
```powershell
$env:VITE_API_BASE_URL="https://<YOUR_API_URL>.onrender.com"
npm run build
firebase deploy --only hosting
```

### Bash / Zsh (macOS/Linux):
```bash
VITE_API_BASE_URL="https://<YOUR_API_URL>.onrender.com" npm run build
firebase deploy --only hosting
```

---

## 4. Live Sanity Check Verification

1. Navigate to `https://ghostlayer-ai.web.app/pilot` in your browser.
2. Fill out and submit a sample workload audit request.
3. Confirm clean redirection to `/pilot/sent?id=<ID>`.
4. Click through to `/report/<ID>` and verify that status displays as **Received**.

---

## 5. Attaching Audit Findings Post-Analysis

Once GhostLayer is run against a real training loop, attach the empirical decision findings via `PATCH`:

```bash
curl -X PATCH https://<YOUR_API_URL>.onrender.com/api/pilot-requests/<ID> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_ADMIN_TOKEN>" \
  -d '{
    "status": "delivered",
    "findings": {
      "summary": "Observed 1.8x throughput headroom with AMP fp16 and batch size 64.",
      "hardwareName": "NVIDIA RTX A2000",
      "precisionSpeedupPct": 80,
      "isEmpiricallyMeasured": true,
      "caveats": "Measured on reference workload; verify loss-shift proxy threshold."
    }
  }'
```

---

## Known Operational Characteristics

* **Render Free Tier Spin-Down:** Free instances spin down after ~15 minutes of inactivity and take 30–60 seconds to wake on subsequent requests.
