"""Exception hierarchy for cascadir.

Every public function validates its inputs up front and raises one of these
exceptions with an *actionable* message (it names the offending column /
condition / threshold and tells the caller how to fix it). This is deliberate:
the method only produces meaningful direction calls on data that satisfies its
contract, so we fail loudly and early rather than silently returning garbage.
"""

from __future__ import annotations


class CascadirError(Exception):
    """Base class for all cascadir errors. Catch this to catch any cascadir failure."""


class DataValidationError(CascadirError):
    """The input AnnData does not satisfy the method's data contract.

    Raised by :func:`cascadir.validate.validate_anndata` (in strict mode) and by
    entry points that re-validate. The message lists *every* blocking issue found,
    not just the first, so the caller can fix them in one pass.
    """


class NotPreprocessedError(CascadirError):
    """The expression matrix is in an ambiguous / unsupported normalization state.

    Raised by :func:`cascadir.preprocess.preprocess` when ``assume="auto"`` cannot
    confidently decide whether ``adata.X`` holds raw counts or log-normalized
    values. The message tells the caller to pass ``assume="raw"`` or
    ``assume="lognorm"`` explicitly.
    """


class InsufficientDataError(CascadirError):
    """Not enough cells / donors / cell types to run a requested step.

    Raised when, e.g., a (condition, cell_type) group has fewer than ``min_cells``
    cells, or fewer than the required number of distinct donors are present. The
    message names the count found and the threshold required.
    """


class SignatureError(CascadirError):
    """A discovered signature is missing, empty, or unusable for cross_asym.

    Raised when a requested condition has no signature, when ``top_n`` exceeds the
    number of available genes, or when two conditions share an identical signature
    (``S_a == S_b``), which collapses the antisymmetric statistic.
    """


class NotFittedError(CascadirError):
    """A :class:`cascadir.pipeline.CascadeDirection` method was called before ``fit``.

    Call ``.fit(adata)`` before ``.direction(...)`` / ``.direction_table(...)``.
    """
