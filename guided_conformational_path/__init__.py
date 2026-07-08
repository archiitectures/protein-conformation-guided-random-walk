"""Protein Conformation Guided-Random Walk prototype package.

The package-level namespace is intentionally sparse in this first version. Only
``ConformationNode`` is re-exported because it is the common object passed
between generators, scorers, graph construction, and path search. Other modules
remain explicit imports while the API is still stabilizing.
"""

from guided_conformational_path.nodes import ConformationNode

__all__ = ["ConformationNode"]
