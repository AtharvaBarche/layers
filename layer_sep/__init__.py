"""Rule-based P&ID layer separation.

The original image is read and never modified. Pipe, instrument, text, and
junction results are written to a dedicated directory.
"""

from layer_sep.pipeline import separate_layers

__all__ = ["separate_layers"]
