Render / Railway deployment notes

Render
- Use a web service with the Dockerfile in repo.
- Environment variables:
  - `GROQ_API_KEY` (optional) — set in Render dashboard as a secret.
  - `PRELOAD_EMBEDDINGS` (optional) — set to `true` to warm embeddings at startup (may increase build time).
  - `PORT` — Render sets automatically; default 8000 used.
- Health check: `HTTP GET /health` should return 200.

Railway
- Add a Docker service using the provided `Dockerfile` and set the same env vars in the Railway project settings.

Notes
- Do NOT commit `.env` or secrets to repo. Use platform secrets.
- For higher availability, run behind a load balancer and increase `--workers` and memory as needed.
