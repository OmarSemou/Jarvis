"""Small versioned SQLite persistence layer for local BMO memory."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Iterable

from .models import MemoryCandidate, MemoryCategory, MemoryConfidence, MemoryEntry, MemorySource


SCHEMA_VERSION = 1


class MemoryStoreError(RuntimeError):
    """The memory database is unavailable or has an unsupported schema."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteMemoryStore:
    """Serialized, short-lived-work SQLite store.

    Construction is side-effect free.  ``initialize`` is the explicit point at
    which the parent directory and database are created.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError("memory database path must be absolute")
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    @property
    def initialized(self) -> bool:
        return self._connection is not None

    def initialize(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(
                    self.path,
                    timeout=5.0,
                    check_same_thread=False,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    connection.close()
                    raise MemoryStoreError(
                        f"memory database schema {version} is newer than supported {SCHEMA_VERSION}"
                    )
                if version == 0:
                    connection.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS memories (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            category TEXT NOT NULL,
                            key TEXT NOT NULL,
                            value TEXT NOT NULL,
                            summary TEXT NOT NULL DEFAULT '',
                            source TEXT NOT NULL,
                            confidence TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            last_used_at TEXT,
                            is_active INTEGER NOT NULL DEFAULT 1,
                            priority INTEGER NOT NULL DEFAULT 0,
                            supersedes_id INTEGER,
                            FOREIGN KEY (supersedes_id) REFERENCES memories(id)
                        );
                        CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
                        CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(category, key, is_active);
                        PRAGMA user_version = 1;
                        """
                    )
                    connection.commit()
                self._connection = connection
            except MemoryStoreError:
                raise
            except (OSError, sqlite3.Error) as exc:
                if "connection" in locals():
                    try:
                        connection.close()
                    except Exception:
                        pass
                raise MemoryStoreError(f"could not initialize memory database: {exc}") from exc

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise MemoryStoreError("memory store has not been initialized")
        return self._connection

    @staticmethod
    def _entry(row: sqlite3.Row) -> MemoryEntry:
        try:
            return MemoryEntry(
                id=int(row["id"]),
                category=MemoryCategory(row["category"]),
                key=str(row["key"]),
                value=str(row["value"]),
                summary=str(row["summary"]),
                source=MemorySource(row["source"]),
                confidence=MemoryConfidence(row["confidence"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                last_used_at=row["last_used_at"],
                is_active=bool(row["is_active"]),
                priority=int(row["priority"]),
                supersedes_id=(int(row["supersedes_id"]) if row["supersedes_id"] is not None else None),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise MemoryStoreError("memory database contains an invalid record") from exc

    def upsert(self, candidate: MemoryCandidate) -> MemoryEntry:
        with self._lock:
            db = self._db()
            now = _now()
            try:
                old = db.execute(
                    "SELECT * FROM memories WHERE category = ? AND key = ? AND is_active = 1 "
                    "ORDER BY id DESC LIMIT 1",
                    (candidate.category.value, candidate.key),
                ).fetchone()
                if old is not None:
                    db.execute("UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?", (now, old["id"]))
                cursor = db.execute(
                    "INSERT INTO memories(category,key,value,summary,source,confidence,created_at,updated_at," 
                    "last_used_at,is_active,priority,supersedes_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        candidate.category.value,
                        candidate.key,
                        candidate.value,
                        candidate.summary,
                        candidate.source.value,
                        candidate.confidence.value,
                        now,
                        now,
                        None,
                        1,
                        candidate.priority,
                        old["id"] if old is not None else None,
                    ),
                )
                db.commit()
                row = db.execute("SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
                if row is None:
                    raise MemoryStoreError("memory insert returned no record")
                return self._entry(row)
            except MemoryStoreError:
                raise
            except sqlite3.Error as exc:
                db.rollback()
                raise MemoryStoreError(f"memory write failed: {exc}") from exc

    def get(self, memory_id: int, *, include_inactive: bool = False) -> MemoryEntry | None:
        with self._lock:
            try:
                query = "SELECT * FROM memories WHERE id = ?"
                params: tuple[object, ...] = (memory_id,)
                if not include_inactive:
                    query += " AND is_active = 1"
                row = self._db().execute(query, params).fetchone()
                return None if row is None else self._entry(row)
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"memory read failed: {exc}") from exc

    def list(self, *, include_inactive: bool = False, limit: int = 100) -> tuple[MemoryEntry, ...]:
        with self._lock:
            try:
                query = "SELECT * FROM memories"
                if not include_inactive:
                    query += " WHERE is_active = 1"
                query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
                rows = self._db().execute(query, (limit,)).fetchall()
                return tuple(self._entry(row) for row in rows)
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"memory list failed: {exc}") from exc

    def deactivate(self, memory_id: int) -> bool:
        with self._lock:
            try:
                cursor = self._db().execute(
                    "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ? AND is_active = 1",
                    (_now(), memory_id),
                )
                self._db().commit()
                return cursor.rowcount == 1
            except sqlite3.Error as exc:
                self._db().rollback()
                raise MemoryStoreError(f"memory forget failed: {exc}") from exc

    def deactivate_all(self) -> int:
        with self._lock:
            try:
                cursor = self._db().execute(
                    "UPDATE memories SET is_active = 0, updated_at = ? WHERE is_active = 1",
                    (_now(),),
                )
                self._db().commit()
                return int(cursor.rowcount)
            except sqlite3.Error as exc:
                self._db().rollback()
                raise MemoryStoreError(f"memory forget-all failed: {exc}") from exc

    def mark_used(self, ids: Iterable[int]) -> None:
        values = tuple(int(value) for value in ids)
        if not values:
            return
        with self._lock:
            try:
                now = _now()
                self._db().executemany(
                    "UPDATE memories SET last_used_at = ? WHERE id = ? AND is_active = 1",
                    ((now, value) for value in values),
                )
                self._db().commit()
            except sqlite3.Error as exc:
                self._db().rollback()
                raise MemoryStoreError(f"memory usage update failed: {exc}") from exc

    def status(self) -> dict[str, object]:
        """Read-only health metadata; values are intentionally not returned."""

        with self._lock:
            try:
                db = self._db()
                active = int(db.execute("SELECT COUNT(*) FROM memories WHERE is_active = 1").fetchone()[0])
                total = int(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
                version = int(db.execute("PRAGMA user_version").fetchone()[0])
                return {"path": str(self.path), "schema_version": version, "active": active, "total": total}
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"memory status failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "SQLiteMemoryStore":
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
