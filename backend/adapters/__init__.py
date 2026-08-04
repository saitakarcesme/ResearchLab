from .autoresearch import KarpathyAutoresearchAdapter
from .base import AdapterContext, ResearchAdapter, ResearchComplete, ResearchFailed
from .huggingface_beyefendi import HuggingFaceBeyefendiBenchmarkAdapter
from .ollama import OllamaBenchmarkAdapter

__all__ = [
    "AdapterContext",
    "HuggingFaceBeyefendiBenchmarkAdapter",
    "KarpathyAutoresearchAdapter",
    "OllamaBenchmarkAdapter",
    "ResearchAdapter",
    "ResearchComplete",
    "ResearchFailed",
]
