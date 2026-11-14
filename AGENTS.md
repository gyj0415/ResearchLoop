# Project working conventions

- Keep project decisions, reference material, evaluation protocols and update logs in `docs/`. Start with `docs/README.md`.
- `docs/` is intentionally local-only and ignored by Git. Do not force-add it or upload archives containing it. A fresh source checkout may not contain this directory.
- Record substantive implementation changes in `docs/CHANGELOG.md`; link the relevant decision and actual verification results. Distinguish planned work, completed implementation and measured outcomes.
- Keep new dependencies, caches, datasets and generated products under this checkout or D:. Local inference uses Ollama; cloud inference stays disabled unless the user changes that requirement.
- Preserve historical runs and memory versions. Never overwrite old results to make a new version look better.
- Tasks A and B have been inspected during development. Use them for diagnostics only. Use a preregistered new task for new transfer claims. The original 80-question final set remains sealed until an explicit final acceptance action.
- Keep evaluation code and evidence sentence identities outside the agent-editable strategy. Both cold and warm agents receive the same deterministic de-duplication, screening and stop rules.
- Do not claim agent or memory gains without matched actual runs and appropriate simple-search / carry-over controls.
