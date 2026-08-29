# BMO Phase 2E memory

Phase 2E adds a deliberately small, local SQLite memory service. It is not
conversation history and it is not a second system prompt. The database is
created only when the application explicitly constructs an enabled
`MemoryService`; importing `jarvis` and running the read-only preflight never
creates it. The default location is ignored runtime data at `data/bmo.db`.

The service stores bounded records with a category, stable key, value, summary,
provenance, confidence, timestamps, active state, and optional replacement
link. Replacing a preference deactivates the previous value rather than
silently creating two active values. Records are inspectable and deactivated
through the CLI:

```powershell
python -m jarvis memory status
python -m jarvis memory list
python -m jarvis memory search voice
python -m jarvis memory show 12
python -m jarvis memory forget 12
python -m jarvis memory forget-all --confirm
```

Interactive text chat exposes the same operations below `/memory`. Forget-all
requires a literal confirmation. Natural-language writes and forget requests
use only the allowlisted structured `remember_memory` and `forget_memory`
tools. Strong explicit commands are recognized deterministically before the
model response round, so a phrase such as “Remember that I prefer pistachio ice
cream” cannot be acknowledged unless its SQLite write has committed. The
provider-neutral tool result is then given to the local model for natural
wording. Less explicit or ambiguous prose remains on the normal structured-tool
path and fails closed if it cannot be safely extracted. Tools never receive
SQL, filesystem paths, arbitrary Python names, or shell access.

The policy rejects secrets and authentication material, sensitive personal
data, transient/debug text, transcript-sized values, hidden prompts, reasoning,
raw audio, and tool traces. Low-confidence model candidates are rejected.
Memory failures are non-critical: conversation, voice, face, robot safety, and
STOP continue, while writes and retrieval fail closed with a concise diagnostic.

Retrieved entries are ranked by deterministic token overlap and bounded by the
configured entry/character limits. Broad persistent-memory questions receive
the bounded active set, while an empty or non-matching query receives an
explicit “No persistent memories were retrieved” signal inside the clearly
marked `<untrusted_remembered_user_context>` data block. Stored text is never
treated as an instruction and never grants robot or safety authority. Session
conversation history is not persistent memory. BMO's immutable identity is
defined by the personality profile and remains available when the memory
database is empty, disabled, deleted, or unavailable. No fictional
Adventure Time memories are seeded.
The active-record cap is also bounded by `memory_max_records` (500 by default).

This phase does not add web lookup, cameras, vision, ESP32 communication, or
physical hardware. The existing simulated robot remains a software test aid,
not hardware safety validation.

The SQLite schema is versioned with `PRAGMA user_version`. A newer schema is
refused rather than reset or destroyed; the current implementation supports
schema version 1 only.
