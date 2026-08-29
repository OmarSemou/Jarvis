"""Controlled, local, persistent BMO memory.

Memory is deliberately separate from conversation history and robot authority.
The package contains no model/provider or hardware dependencies.
"""

from .models import (
    MemoryCandidate,
    MemoryCategory,
    MemoryConfidence,
    MemoryEntry,
    MemorySource,
)
from .policy import MemoryPolicy, MemoryPolicyError
from .service import MemoryActionResult, MemoryService
from .sqlite_store import MemoryStoreError, SQLiteMemoryStore
from .tools import MEMORY_TOOL_DEFINITIONS, MemoryToolExecutor, MemoryToolRegistry
from .intent import ExplicitMemoryRequest, is_persistent_memory_query, recognize_explicit_memory_request

__all__ = [
    "MemoryActionResult",
    "MemoryCandidate",
    "MemoryCategory",
    "MemoryConfidence",
    "MemoryEntry",
    "MemoryPolicy",
    "MemoryPolicyError",
    "MemoryService",
    "MemorySource",
    "MemoryStoreError",
    "SQLiteMemoryStore",
    "MEMORY_TOOL_DEFINITIONS",
    "MemoryToolExecutor",
    "MemoryToolRegistry",
    "ExplicitMemoryRequest",
    "recognize_explicit_memory_request",
    "is_persistent_memory_query",
]
