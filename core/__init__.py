from .memory_manager import MemoryManager
from .message import Message

__all__ = ["MemoryManager", "Message", "Orchestrator"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from .orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
