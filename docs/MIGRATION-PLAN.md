# Migration Plan — Remove DB + Dashboards from collab-environment

## Context

The database layer (`collab_env/data/db/`) and the two Panel dashboards (`collab_env/dashboard/`) are being moved to the [collab-data](https://github.com/anthropics/collab-data) repo, where they belong conceptually (data storage, querying, and exploration — not environment/sim/gnn/tracking). PR anthropics/collab-environment#73 introduced the analysis dashboard + db layer; the GCS browser predates it.

This document covers **only the cleanup of this repo**. It is **Phase 2**, and only runs after the collab-data import PR has merged and been verified end-to-end. For the corresponding Phase 1 plan see `docs/MIGRATION-PLAN.md` in the collab-data repo (branch `migrate-db-and-dashboards`).

## Decisions captured up front

- `collab_env/data/file_utils.py` and `collab_env/data/gcs_utils.py` **stay in this repo**. Roughly 12 modules across `sim/`, `gnn/`, and `tracking/` import from them; we are not rewriting those import sites and not taking a cross-repo dependency on `collab-data`. Accept that these two files are now duplicated between repos and may drift.
- One PR for the entire cleanup.
- No history rewrite in this repo — just `git rm` the moved tree.

## Preconditions before starting

- collab-data PR `migrate-db-and-dashboards` is merged to `master`.
- A clean checkout of collab-data on a fresh venv passes:
  - `pip install -e .[dev]`
  - imports for `collab_data.db.*`, `collab_data.data_dashboard.app`, `collab_data.analysis_dashboard.spatial_analysis_gui`
  - `pytest tests/db tests/analysis_dashboard`
  - launching both `scripts/data_dashboard.sh` and `scripts/analysis_dashboard.sh` and clicking through widgets / video viewer
- A note in this repo's PR description pointing at the corresponding collab-data merge commit.

## Branch

Cut a new branch off `main` (e.g. `cleanup-db-and-dashboards`).

## Files and directories to delete

### Python packages

- `collab_env/data/db/` — entire subpackage (config, db_loader, init_database, query_backend, schema/, queries/)
- `collab_env/dashboard/` — entire package (app, cli, dashboard_app, file_viewers, persistent_video_server, rclone_client, session_manager, spatial_analysis_app, spatial_analysis_gui, analysis_widgets.yaml, widgets/, templates/, static/, utils/simulation_loader.py)

`collab_env/data/__init__.py`, `file_utils.py`, `gcs_utils.py` — **keep**.

### Tests

- `tests/db/` — entire directory
- `tests/dashboard/` — entire directory
- `tests/test_dashboard.py`
- `tests/conftest.py` — review: keep the parts unrelated to db/dashboard skipping; drop any DB-specific collect_ignore logic

### Scripts

- `scripts/dev_dashboard.sh`
- `scripts/analysis_dashboard.sh`
- `scripts/deploy/` — entire directory (config.sh, cloudbuild.yaml, Dockerfile.dashboard, build_and_deploy.sh, setup_cloud_sql.sh, start_proxy.sh, README.md)

### Docs

- `docs/data/db/` — entire directory
- `docs/dashboard/` — entire directory (README.md, DATA_DASHBOARD_README.md, CLOUD_SETUP.md, spatial_analysis.md, grafana/)
- `docs/data/gcloud_*.ipynb` — these moved to collab-data; remove from here too
- Leave `docs/data/` itself only if other files remain in it (otherwise remove the directory)

### Top-level

- `requirements-db.txt` — delete

## Files to edit

### `pyproject.toml`

- Remove from `[tool.setuptools] packages`:
  - `collab_env.data.db`
  - `collab_env.dashboard`
- Remove from `[tool.setuptools.package-data]`:
  - `"collab_env.data.db" = [...]`
  - `"collab_env.dashboard" = [...]`
- Remove the entire `db` and `db-dashboard` optional dependency groups from `[project.optional-dependencies]`
- Edit the `dev` optional group to drop the DB/dashboard transitive deps (`aiosql`, `sqlalchemy`, `psycopg2`, `panel`, `holoviews`, `bokeh`, `plotly`, `duckdb*`, `pyarrow` if not needed elsewhere)

After editing, run `pip install -e .[dev]` in a clean venv and verify the resolve has shrunk.

### `README.rst`

- Drop the **Dashboard** section entirely
- Remove the rclone / ffmpeg / exiftool setup steps from the prerequisites section, **unless** any of these are still required by tracking or another module (verify before deleting — `exiftool` in particular may still be used by tracking; grep before removing)
- Update the docs links section to remove the now-deleted `docs/dashboard/README.md` link

### `Makefile`

- Currently has no dashboard-specific targets per inventory; verify with a grep and remove anything that does reference dashboards/db.

### `.github/workflows/test.yml`

- Currently runs `./scripts/test.sh`, which discovers all of `tests/`. After the test directories are deleted no workflow change is required, but verify nothing references the deleted paths or env vars (`POSTGRES_*`, `DB_BACKEND`, etc.).

### `scripts/test.sh`, `scripts/lint.sh`, `scripts/clean.sh`

- Verify none of them reference the deleted paths. They are general-purpose, so likely no edits needed.

## Critical files to read while executing

- `pyproject.toml` — to know exactly which optional groups and package_data entries to prune
- `README.rst` — for the dashboard section and prerequisites
- `tests/conftest.py` — to understand the SKIP_GCS_TESTS hook and not break GCS test gating
- `scripts/test.sh` and `.github/workflows/test.yml` — to confirm no hardcoded references to the deleted directories

## Cross-dependency check (must pass before deletion)

Before deleting, grep the **rest of** `collab_env/` for any remaining imports from the dirs being removed:

```bash
rg "from collab_env\.(data\.db|dashboard)" collab_env/
rg "import collab_env\.(data\.db|dashboard)" collab_env/
```

Both should return zero hits. (Per inventory: only the dashboard itself imports from `collab_env.data.db`, and nothing else in the repo imports from `collab_env.dashboard`.)

Also verify the surviving `collab_env/data/file_utils.py` imports are still satisfied:

```bash
rg "from collab_env\.data\.file_utils" collab_env/
rg "from collab_env\.data\.gcs_utils" collab_env/
```

These should still resolve (the files remain). The ~12 sim/gnn/tracking import sites identified during exploration must continue to work — this is the whole reason we kept those two files.

## Verification

1. `pip install -e .[dev]` in a fresh venv succeeds and the resolve no longer pulls in panel / holoviz / sqlalchemy / psycopg2 / aiosql / duckdb.
2. `python -c "from collab_env.data.file_utils import expand_path, get_project_root; from collab_env.data.gcs_utils import GCSClient"` still works.
3. Smoke import the modules that were the heaviest file_utils consumers:

   ```bash
   python -c "import collab_env.sim.boids.run_simulator"
   python -c "import collab_env.gnn.train"
   python -c "import collab_env.gnn.gnn_3D.train_3DGNN"
   ```

4. `./scripts/test.sh` (with `SKIP_GCS_TESTS=1`) passes. The test count should drop by exactly the deleted db + dashboard tests; sim / gnn / tracking tests must still pass.
5. `./scripts/lint.sh` passes.
6. CI: open PR, confirm `test.yml`, `lint.yml`, `test_notebooks.yml`, `lint_notebooks.yml` are all green.
7. Verify the PR description links to the merged collab-data import PR.
