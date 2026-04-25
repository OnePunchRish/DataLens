# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## CSV Analytics Dashboard

A standalone Flask web app at `artifacts/csv-dashboard/`.

### Stack
- **Python 3.11** + Flask
- **Pandas** for CSV parsing and data processing
- **Plotly** for interactive charts (bar, pie, line)
- Jinja2 HTML templates (`templates/index.html`, `templates/dashboard.html`)

### Structure
```
artifacts/csv-dashboard/
├── app.py                    # Flask routes and chart generation
├── templates/
│   ├── index.html            # Upload page
│   └── dashboard.html        # Data table + charts dashboard
└── uploads/                  # Uploaded CSV files (temp storage)
```

### Workflow
- **Start application**: `cd artifacts/csv-dashboard && PORT=5000 python app.py`
- Serves on port 5000

### Features
- CSV file upload (up to 16 MB)
- Dataset table (first 100 rows)
- Auto-generated bar, pie, and line charts based on column types
- Summary statistics for numeric columns
- Column schema browser with type badges
- Missing values report
