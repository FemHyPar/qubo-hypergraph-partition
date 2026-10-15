"""Public interface; the optional catalyst backend is a compiled extension."""
from .api import partition
from .model import Hypergraph, score

__version__ = "0.1.0a4"
__all__ = ["Hypergraph", "partition", "score"]
