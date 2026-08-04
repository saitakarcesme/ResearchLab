from .autoresearch import KarpathyAutoresearchAdapter
from .base import AdapterContext, ResearchAdapter, ResearchComplete, ResearchFailed
from .ollama import OllamaBenchmarkAdapter

__all__ = [
    "AdapterContext",
    "KarpathyAutoresearchAdapter",
    "OllamaBenchmarkAdapter",
    "ResearchAdapter",
    "ResearchComplete",
    "ResearchFailed",
]
