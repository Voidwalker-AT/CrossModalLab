# Minimal dashboard dependency plan

Scope: read-only stored-artifact dashboard for the October 28 university demonstration. No training, inference, model downloads, global installs, or changes to the Python ML environment.

Backend: existing Python 3.11+ standard library HTTP/JSON server and project symbolic solver. No FastAPI/Pydantic/uvicorn installation is needed for this foundation. The API contract can later be hosted by FastAPI without changing the data adapters.

Frontend: React and React DOM 19.3.1, Vite 8.3.5, TypeScript 7.0.2, React/React DOM type definitions 19.3.0. Versions were checked against npm registry metadata on 2026-10-23. Existing Node 24.15.0 satisfies Vite's declared Node requirement. No chart/graph framework, component library, CDN, analytics SDK or GPU client is needed; charts and graphs use SVG.

Approval requested for **project-local installation of these six direct packages and their locked transitive dependencies**, using the existing Node/npm runtime. Proposed command, run inside `dashboard/frontend/`:

```powershell
$env:npm_config_cache = Join-Path (Resolve-Path ../..) '.cache/npm'
npm install --ignore-scripts --no-audit --no-fund
```

This writes `node_modules/` and a reproducible package-lock within the workspace and uses the workspace npm cache. Lifecycle scripts are disabled. No existing Python dependency or lockfile is changed. `package.json` is reviewable before approval. The resulting lockfile is inspected and committed; modules/caches remain ignored. The user explicitly approved this installation on 2026-10-23. Execution used Node 24.15.0 / npm 11.12.1, installed 23 packages, and the production build passed. No global package was installed.

Verification: TypeScript check, Vite production build, CPU API tests, request rejection/path containment tests, browser navigation/filter/sample/graph checks, no GPU loading, and the existing repository quality gate. Ordinary Python serves the built frontend and artifacts in one local origin; ULVO is optional for contributors.
