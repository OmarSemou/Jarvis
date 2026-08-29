"""Compatibility exports for the memory storage boundary."""

from .sqlite_store import MemoryStoreError, SCHEMA_VERSION, SQLiteMemoryStore

MemoryStore = SQLiteMemoryStore

__all__ = ["MemoryStore", "MemoryStoreError", "SCHEMA_VERSION", "SQLiteMemoryStore"]
