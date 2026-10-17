"""Public interface; the optional catalyst backend is a compiled extension."""
from .api import partition,partition_fp32
from .model import Hypergraph, score

__version__ = "0.1.0a5"
__all__ = ["Hypergraph", "partition", "partition_fp32", "score"]
