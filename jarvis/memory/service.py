"""Application service coordinating policy, storage, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import MemoryCandidate, MemoryEntry
from .policy import MemoryPolicy, MemoryPolicyError
from .retrieval import rank_entries, render_context, render_empty_context
from .intent import is_persistent_memory_query
from .sqlite_store import MemoryStoreError, SQLiteMemoryStore


@dataclass(frozen=True, slots=True)
class MemoryActionResult:
    success: bool
    message: str
    entry: MemoryEntry | None = None
    reason: str | None = None


class MemoryService:
    """Non-critical memory service. Failures never become authority failures."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        policy: MemoryPolicy | None = None,
        *,
        enabled: bool = True,
        logger: Callable[[str], None] | None = None,
        debug: bool = False,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.enabled = bool(enabled)
        self._logger = logger
        self._debug = bool(debug)
        self._initialized = False
        if self.enabled:
            self.initialize()

    def initialize(self) -> None:
        if not self.enabled or self._initialized:
            return
        try:
            self.store.initialize()
            self._initialized = True
        except MemoryStoreError:
            # Keep the service available for chat; writes/retrieval fail closed.
            self._initialized = False
            self._log("unavailable reason=memory_store_initialization_failed")
            return
        if self._debug:
            self._log(f"database={self.store.path}")

    open = initialize

    def _log(self, message: str) -> None:
        if self._logger is not None:
            try:
                self._logger(f"[MEMORY] {message}")
            except Exception:
                pass

    def _unavailable(self) -> MemoryActionResult:
        return MemoryActionResult(False, "Memory is unavailable; conversation continues without persistence.", reason="memory_unavailable")

    @property
    def available(self) -> bool:
        return self.enabled and self._initialized

    @property
    def database_path(self) -> Path:
        return self.store.path

    def remember(self, candidate: MemoryCandidate, *, explicit: bool = False) -> MemoryActionResult:
        if not self.enabled:
            return MemoryActionResult(False, "Memory is disabled.", reason="memory_disabled")
        try:
            accepted = self.policy.validate(candidate, explicit=explicit)
            if not self._initialized:
                return self._unavailable()
            active_records = self.store.list(limit=self.policy.max_records + 1)
            replacing_existing = any(
                entry.category is accepted.category and entry.key == accepted.key
                for entry in active_records
            )
            if len(active_records) >= self.policy.max_records and not replacing_existing:
                return MemoryActionResult(False, "Memory capacity is full; remove an old memory first.", reason="memory_limit")
            entry = self.store.upsert(accepted)
            action = "updated" if replacing_existing else "created"
            self._log(f"{action} id={entry.id} category={entry.category.value} key={entry.key}")
            return MemoryActionResult(True, f"Remembered {entry.key}.", entry=entry)
        except MemoryPolicyError as exc:
            self._log(f"rejected reason={exc.reason}")
            return MemoryActionResult(False, f"I did not store that ({exc.reason}).", reason=exc.reason)
        except (MemoryStoreError, OSError):
            self._log("rejected reason=memory_write_failed")
            return self._unavailable()

    def forget(self, memory_id: int) -> MemoryActionResult:
        if not self.enabled:
            return MemoryActionResult(False, "Memory is disabled.", reason="memory_disabled")
        if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id <= 0:
            return MemoryActionResult(False, "Memory id must be a positive integer.", reason="invalid_id")
        if not self._initialized:
            return self._unavailable()
        try:
            if not self.store.deactivate(memory_id):
                self._log("rejected reason=not_found")
                return MemoryActionResult(False, "No active memory has that id.", reason="not_found")
            self._log(f"forgotten id={memory_id}")
            return MemoryActionResult(True, f"Forgot memory {memory_id}.")
        except MemoryStoreError:
            self._log("rejected reason=memory_forget_failed")
            return self._unavailable()

    def forget_query(self, query: str) -> MemoryActionResult:
        if not isinstance(query, str) or not query.strip():
            return MemoryActionResult(False, "Describe one memory to forget.", reason="empty_query")
        matches = self.search(query)
        if len(matches) != 1:
            reason = "not_found" if not matches else "ambiguous"
            self._log(f"rejected reason={reason}")
            return MemoryActionResult(False, "I need one unambiguous memory to forget.", reason=reason)
        return self.forget(matches[0].id)

    def list(self, *, include_inactive: bool = False, limit: int = 100) -> tuple[MemoryEntry, ...]:
        if not self.enabled or not self._initialized:
            return ()
        try:
            return self.store.list(include_inactive=include_inactive, limit=max(1, min(limit, 500)))
        except MemoryStoreError:
            return ()

    def show(self, memory_id: int) -> MemoryEntry | None:
        if not self.enabled or not self._initialized:
            return None
        try:
            return self.store.get(memory_id)
        except MemoryStoreError:
            return None

    def search(self, query: str) -> tuple[MemoryEntry, ...]:
        return rank_entries(query, self.list(), limit=100)

    def retrieve_context(self, query: str) -> str:
        memory_query = is_persistent_memory_query(query)
        if not self.enabled or not self._initialized:
            return render_empty_context(reason="Persistent memory is unavailable for this request.") if memory_query else ""
        try:
            entries = self.store.list(limit=500)
            # Broad memory questions should receive the bounded active set, not
            # an empty result caused by the words "what do you remember" not
            # overlapping a stored key/value.
            selected = (
                entries[: self.policy.max_context_entries]
                if memory_query and not rank_entries(query, entries, limit=1)
                else rank_entries(query, entries, limit=self.policy.max_context_entries)
            )
            context = render_context(selected, max_chars=self.policy.max_context_chars)
            if selected and context:
                self.store.mark_used(entry.id for entry in selected if context)
            elif memory_query:
                return render_empty_context(max_chars=self.policy.max_context_chars)
            return context if selected else ""
        except MemoryStoreError:
            return render_empty_context(reason="Persistent memory could not be read for this request.") if memory_query else ""

    def status(self) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "available": False, "reason": "memory_disabled", "path": str(self.store.path)}
        if not self._initialized:
            return {"enabled": True, "available": False, "reason": "memory_unavailable", "path": str(self.store.path)}
        try:
            return {"enabled": True, "available": True, **self.store.status()}
        except MemoryStoreError:
            return {"enabled": True, "available": False, "reason": "memory_unavailable", "path": str(self.store.path)}

    def forget_all(self) -> MemoryActionResult:
        if not self.enabled:
            return MemoryActionResult(False, "Memory is disabled.", reason="memory_disabled")
        if not self._initialized:
            return self._unavailable()
        try:
            count = self.store.deactivate_all()
            self._log(f"deactivated_all count={count}")
            return MemoryActionResult(True, f"Forgot {count} memories.")
        except MemoryStoreError:
            return self._unavailable()

    def close(self) -> None:
        self.store.close()
        self._initialized = False
