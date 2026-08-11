# Agent Instructions

## Skills & Capabilities

- **Check and utilize available skills**: Whenever an available skill relates to your task (e.g., framework-specific workflows, testing patterns, API integrations, or customization tools), inspect and activate the skill's instructions before proceeding.
- **Do not reinvent existing tooling**: Rely on project scripts and configured skill guidelines for consistency across agents.
- **Superpowers Planning & Coordination (`docs/superpowers/`)**: The `docs/superpowers/` (or `/superpowers`) directory is specifically reserved for agent skills to plan, create design specifications (`docs/superpowers/specs/`), and coordinate task execution (`docs/superpowers/plans/`). Use these structures for multi-step agentic workflows.

---

## Documentation

**Always read before starting any task:**
1. `README.md` - project overview, features, structure. Check `docs/README.md` for map of technical details.
2. `docs/README.md` - full documentation index and current state of the project.

This order matters: README gives the big picture fast, `docs/README.md` maps every topic to its file. Then read the doc file(s) relevant to the area you are working in - fully, not just grepped. The doc will tell you where things live (backend, components, utils, views, etc.) so you can go straight to the right file instead of searching.

**Keep documentation up to date.** When you make changes:
- If the change affects features, structure, or project-level info - update `README.md`
- If the change affects a topic covered in `docs/` - update the relevant doc file and `docs/README.md`
- If the change affects in-app documentation - update `assets/docs_reference.json` and `views/documentation.py`
- If a change introduces something not covered by any existing doc - add it

The relevant doc file is usually obvious from `docs/README.md`.

---

## Mandatory Pre-Response Protocol

Before finalizing any response after modifying or adding code, you MUST execute this check:
1. **Inspect Changes:** Review all modified files (e.g. via `git status` / diff inspection).
2. **Documentation Audit:** Cross-reference changes against `docs/` and `README.md`:
   - If CLI options, defaults, or prompts changed → update `docs/11-headless-cli.md`
   - If models, preprocessing, or constraints changed → update `docs/03-model-zoo-and-preprocessing.md` / `docs/04-benchmark-configuration.md`
   - If file structure or project layout changed → update `README.md` and `docs/README.md`
3. **Atomic Updates:** Apply all required documentation updates in the **same turn** as the code changes, before presenting your response to the user.

---

## Git Commit Workflow

- Only provide commit commands when explicitly asked
- Before writing a commit message, always run `git diff` to check what actually changed
- Do NOT run the commit command directly - provide it as copyable text
- Do NOT include `Co-Authored-By` lines
- Do NOT use heredoc/EOF syntax
- Use `git add` and `git commit` as separate commands
- Format: `git commit -m "subject\n\n- bullet\n- bullet"`
- Subject: imperative mood, lowercase start, no period