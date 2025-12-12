# Claude Code Instructions

## Python Environment

**ALWAYS use `.venv-310` virtual environment for this project.**

When running Python commands, use:
```bash
source .venv-310/bin/activate && python <command>
```

Never use system `python` or `python3` directly without activating the virtual environment first.

## Example Commands

```bash
# Running training
source .venv-310/bin/activate && python -m collab_env.gnn.interaction_particles.run_training --help

# Running any Python script
source .venv-310/bin/activate && python <script>.py
```
