# Development Rules for mistral-text-to-spech

This file provides guidance to AI agents and AI-assisted development tools when working with this project. This includes Claude Code, Cursor IDE, GitHub Copilot, Windsurf, and any other AI coding assistants.

## General Coding Principles
- **Fail fast — never swallow errors.** Always propagate errors and exit with code 1 immediately. No silent fallbacks, no `|| true`, no ignored return codes.
- **Never assume any default values anywhere.** Check for required values explicitly and exit 1 if something is missing. Default values mask underlying issues and make them hard to debug.
- **Never suppress checks with annotations.** Fix the underlying issue instead. No `# noqa`, `# type: ignore`, `# nosec`, `@pytest.mark.filterwarnings`, or any other mechanism that silences a checker.
- Always be explicit about values, paths, and configurations
- If a value is not provided, raise an error — never silently fall back to a default

## Git Commit Guidelines

**IMPORTANT:** When creating git commits in this repository:
- **NEVER include AI attribution in commit messages**
- **NEVER add "Generated with [AI tool name]" or similar phrases**
- **NEVER add "Co-Authored-By: [AI name]" or similar attribution**
- **NEVER run `git add -A` or `git add .` - always stage files explicitly**
- Keep commit messages professional and focused on the changes made
- Commit messages should describe what changed and why, without mentioning AI assistance
- **ALWAYS run `git push` after creating a commit to push changes to the remote repository**
- **NEVER use `git -C <path>`** — always run git commands from the project root directory

## Testing
- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `just chat` after modifications

## Python Execution Rules
- Python code must be executed **only** via `uv run ...`
  - Example: `uv run src/main.py`
  - **Never** use: `python src/main.py` or `python3 src/main.py`
- The virtual environment must be created and updated **only** via `uv sync`
  - **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies must be managed through `uv` and declared in `pyproject.toml`

## Justfile Conventions
- **Use `printf` for colored or formatted output** — never `echo` with ANSI escape sequences, as some terminals won't render colors with `echo`. Plain `echo ""` is acceptable only for blank-line spacing.
- **Add an empty `@echo ""` line before and after each target's command block** to visually separate output between targets.
- **The `help` target must be a dedicated recipe** with manually written `printf` lines that group related commands and order them by typical execution flow (setup → run → code quality → testing). Never use `just --list`.
- **The default target (`_default`) must call `just help`.**
- **Every target must end with a clear status message**: green (`\033[32m`) on success, red (`\033[31m`) on failure with `exit 1`.
- **Composite targets (e.g. `ci`) must fail fast**: use `set -e` or `&&` chaining.
- All Python execution in the justfile uses `uv run`, never `python` directly
- Use `just init` to set up the project
- Use `just chat` to run the interactive chat
- Use `just serve` to start the FastAPI TTS server (foreground)
- Use `just stop` to stop the running server
- Use `just status` to check if the server is running
- Use `just destroy` to remove the virtual environment
- Use `just help` to see all available recipes with descriptions
- Use `just` (with no arguments) to show help
- Use `just ci` to run all validation checks (verbose)
- Use `just ci-quiet` to run all validation checks (silent, fail-fast)

## Project Structure
- All source code lives in `src/`
- Test scripts and utilities go in `scripts/`
- **Input data is organized**: `data/input/`
- **Output data is organized**: `data/output/`
- **Never create Python files in the project root directory**
  - Wrong: `./test.py`, `./helper.py`
  - Correct: `./src/helper.py`, `./scripts/test.py`

## Error Handling
- Fail fast — stop immediately on the first error, never continue past failures
- Never catch and silently ignore exceptions
- Raise exceptions with clear messages for missing or invalid data
- Exit with code 1 if any operation fails, 0 if all succeeded

## Optimization
- **Skip processing if output already exists** - Don't reprocess unnecessarily
- Check if output file exists before starting expensive operations
- Track skipped items separately in summary reports
- Allow users to force reprocessing by deleting output files

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:d4f96305 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- END BEADS INTEGRATION -->
